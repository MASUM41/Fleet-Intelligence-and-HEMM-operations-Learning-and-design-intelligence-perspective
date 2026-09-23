# LaTeX Paper Skeleton — Fleet Intelligence & HEMM Operations

A ready-to-fill IEEE-style paper layout for the project, with every section
pre-structured around the actual system (simulator → data pipeline → ETA
models → RL dispatch shootout). All `TODO` comments mark where your content goes.

## Fastest way to compile: Overleaf (no install)

1. Go to Overleaf → **New Project → Upload Project**
2. Select this zip (`hemm_fleet_paper_latex.zip`)
3. It compiles immediately (main document = `main.tex`)

## Local compile (MiKTeX / TeX Live)

```powershell
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or one-shot: `latexmk -pdf main`

> If `IEEEtran.cls` is missing, install the `ieeetran` package from your TeX
> package manager, or uncomment the `article` fallback lines at the top of `main.tex`.

## Structure

```
paper/
├── main.tex                     # document shell: class, packages, title, abstract
├── sections/
│   ├── 01_introduction.tex      # motivation + contribution bullets
│   ├── 02_related_work.tex      # 4 suggested strands
│   ├── 03_simulation_platform.tex # map, 3-tier decision stack, truck physics, params table
│   ├── 04_data_pipeline.tex     # logging → merge → KPI extraction, trips.csv schema
│   ├── 05_eta_prediction.tex    # sklearn → graph features → GraphSAGE
│   ├── 06_dispatch_rl.tex       # surrogate env, A/B/C/D table, fairness contract, baseline ladder
│   ├── 07_results.tex           # current 5-seed numbers + 4 wired figures
│   ├── 08_discussion.tex        # headroom argument, limitations, threats, future work
│   └── 09_conclusion.tex
├── references.bib               # 4 real starter entries (DQN, Double DQN, GraphSAGE, Sutton&Barto)
└── figures/                     # pre-wired: learning_curves, 6_vs_eta_min (queue distance-normalised),
                                 # 5_seed_spread, 1_reward_bar, 2_radar_profile, 4_metric_heatmap
```

## Notes

- Numbers in `07_results.tex` reflect the **current 5-seed shootout results**
  (C per-mine best mean −267.4; all methods beat ETA_MIN by +5.2…+10.7 % reward).
  Update if you retrain.
- Figures referenced in the text are already in `figures/` — add more from
  `Simulation/analysis/rl/graphs/` as needed.
- The baseline framing (ETA_MIN = primary, GREEDY_NEAREST = oracle ceiling)
  matches the updated `rl_experiments` code and README.
