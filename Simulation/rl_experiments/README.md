# RL Agent-Architecture Shootout

## Setup on a fresh PC (e.g. lab PC with RTX 4060)

```powershell
cd Simulation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # numpy, pandas, matplotlib, torch
# verify GPU is visible (optional):
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Only 4 third-party packages are needed (`numpy pandas matplotlib torch`);
`pygame` is NOT required — the surrogate env is headless.

Which RL agent framing is right for haul-truck dispatch? This suite trains all
four candidates under an **identical fairness contract** and compares them
against 6 hand-built baseline policies.

| Method | Code | Agent(s) | Action | Observation |
|---|---|---|---|---|
| **B** one-for-all | `b` | 1 central DDQN | mine (9) | global fleet state (49) |
| **A** per-truck   | `a` | N independent DDQNs | mine (9) | local per-truck view (38) |
| **C** per-mine    | `c` | 9 bidder DDQNs | bid level (5) → buyer rule picks mine | per-mine view (7) |
| **D** universal   | `d` | 1 central DDQN | mine × dump — logged pairs only (54) | global + dump state (55) |

**Fairness contract:** same env (`DispatchEnv` on the calendar simulator),
same reward (`r = -trip_duration/100`), same 20-config eval suite
(fleet 4–12 × coal scale 0.6–2.0), same budget, same hyperparameters
(Double DQN, γ=0.97, replay 50k, n-step 3, ε 1.0→0.05, target sync 500).
Only the decision architecture changes.

---

## Commands (run from `Simulation/`)

### 0. Smoke test (~30 s each)
```powershell
python -m rl_experiments.run --method b --seed 0 --trips 1500 --eval-every 500 --eval-configs 4
python -m rl_experiments.run --method baselines --eval-configs 4
python -m rl_experiments.report
```

### 1. Reference floor: 6 hand baselines (no learning, ~30 s)
```powershell
python -m rl_experiments.run --method baselines --eval-configs 20
```

### 2. Single training run (one method, one seed)
```powershell
python -m rl_experiments.run --method b --seed 0 --trips 100000 --eval-every 10000
python -m rl_experiments.run --method d --seed 0 --trips 100000 --eval-every 10000 --device cuda
```

### 3. FULL SHOOTOUT — 4 methods × 5 seeds = 20 runs, in parallel
```powershell
# CPU-only machine (auto-detects; ~2-4 hrs with workers=6)
python -m rl_experiments.run --method all --seeds 5 --trips 100000 --eval-every 10000 --workers 6

# With RTX 4060 (auto-detects cuda; ~40-80 min with workers=6-8)
python -m rl_experiments.run --method all --seeds 5 --trips 100000 --eval-every 10000 --workers 8 --device cuda
```

### 4. Budget-sensitivity bonus (tests H4: does D catch up with more trips?)
```powershell
python -m rl_experiments.run --method d --seed 0 --trips 200000 --eval-every 10000
```

### 5. Report (tables + learning curves + hypothesis check)
```powershell
python -m rl_experiments.report
```

---

## Outputs (all in `analysis/rl/`)

| File | Content |
|---|---|
| `{method}_s{seed}_curve.csv` | eval metrics every `--eval-every` trips |
| `baselines_eval.csv` | per-config metrics for the 6 hand policies |
| `learning_curves.png` | reward & duration curves, mean ± seed-range, baseline lines |
| console tables | final comparison + hypothesis check (H1–H4) |

## Method notes / known behaviors

- **A (per-truck)**: proper per-truck MDP — a truck's transition completes when
  *it* pops again (not the next global event). Expect slow, oscillating
  learning (non-stationarity). This is hypothesis H2 being tested, not a bug.
- **C (per-mine)**: mines bid a level in `{-80..80 s}`; the requesting truck
  buys `argmin(travel + wait + cycle + bid)`. One rotating core learns per
  event → gradient-update count equals B (fairness + speed).
- **D (universal)**: joint actions are **masked to logged (mine, dump) pairs**
  — graph-estimated pairs are used for travel-time lookups but not as actions.
  (Without this mask, D reward-hacks optimistic unseen-pair times.)
- Env fidelity: logged pairs use real medians; unseen pairs use Dijkstra
  shortest-path / calibrated effective speed (see `env.RoadNet`).

## File map
```
rl_experiments/
  env.py        DispatchEnv (event-driven MDP) + RoadNet (graph fallback)
  nets.py       QNet/Dueling, n-step replay, DDQNCore (pure PyTorch)
  baselines.py  6 hand policies (factories)
  run.py        train loops B/A/C/D + eval harness + CLI + parallel seeds
  report.py     aggregation, curves, hypothesis check
```
