"""
generate_assets.py - Build map_cache.pkl and waypoints.pkl for the simulator.

Run from Simulation/:
    python generate_assets.py
"""

import heapq
import itertools
import os
import pickle

import numpy as np

from Map import map_loader as map_data

MAP_DIR = os.path.join("Map")
POINTS_PER_SEGMENT = 20


def build_weighted_graph(nodes, edges):
    graph = {name: [] for name in nodes}
    for edge in edges:
        if len(edge) < 2:
            continue
        n1, n2 = edge[0], edge[1]
        if n1 not in nodes or n2 not in nodes:
            continue
        dist = float(np.linalg.norm(nodes[n1] - nodes[n2]))
        graph[n1].append((n2, dist))
        graph[n2].append((n1, dist))
    return graph


def a_star_pathfinding(graph, start_name, goal_name):
    if start_name not in graph or goal_name not in graph:
        return []
    open_set = [(0.0, start_name)]
    came_from = {}
    g_score = {name: float("inf") for name in graph}
    g_score[start_name] = 0.0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_name:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start_name)
            return list(reversed(path))

        for neighbor, weight in graph[current]:
            tentative = g_score[current] + weight
            if tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heapq.heappush(open_set, (tentative, neighbor))
    return []


def catmull_rom_point(t, p0, p1, p2, p3):
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t ** 2)
        + (-p0 + 3 * p1 - 3 * p2 + p3) * (t ** 3)
    )


def generate_curvy_path(node_list):
    if len(node_list) < 2:
        return []
    padded = [node_list[0]] + node_list + [node_list[-1]]
    waypoints = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i : i + 4]
        if i == 0:
            waypoints.append(p1)
        for j in range(1, POINTS_PER_SEGMENT + 1):
            t = j / float(POINTS_PER_SEGMENT)
            waypoints.append(catmull_rom_point(t, p0, p1, p2, p3))
    return waypoints


def generate_waypoints():
    waypoints_data = {}
    for chain in map_data.VISUAL_ROAD_CHAINS:
        chain_key = tuple(chain)
        coords = [map_data.NODES[name] for name in chain_key if name in map_data.NODES]
        if len(coords) < 2:
            continue
        waypoints_data[chain_key] = generate_curvy_path(coords)
    return waypoints_data


def main():
    print("=== Generate Simulation Assets ===\n")

    if not map_data.NODES:
        raise RuntimeError("Map data not loaded. Check Map/map_data.json.")

    print(f"Nodes: {len(map_data.NODES)}  Edges: {len(map_data.EDGES)}")
    print("Building road graph...")
    road_graph = build_weighted_graph(map_data.NODES, map_data.EDGES)

    print("Caching zone routes...")
    fuel_zones = getattr(map_data, "FUEL_ZONES", [])
    all_targets = list(set(map_data.LOAD_ZONES + map_data.DUMP_ZONES + fuel_zones))
    route_cache = {}
    for start, end in itertools.permutations(all_targets, 2):
        path = a_star_pathfinding(road_graph, start, end)
        if path:
            route_cache[(start, end)] = path

    cache_path = os.path.join(MAP_DIR, "map_cache.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"road_graph": road_graph, "route_cache": route_cache}, f)
    print(f"Saved: {cache_path}  ({len(road_graph)} nodes, {len(route_cache)} routes)")

    print("Generating waypoints...")
    waypoints = generate_waypoints()
    wp_path = os.path.join(MAP_DIR, "waypoints.pkl")
    with open(wp_path, "wb") as f:
        pickle.dump(waypoints, f)
    print(f"Saved: {wp_path}  ({len(waypoints)} chains)")

    print("\nDone. You can run: python main.py")


if __name__ == "__main__":
    main()
