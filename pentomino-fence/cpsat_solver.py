#!/usr/bin/env python3
"""
CP-SAT maximizer for enclosed area using pentominoes F, L, N, T, Z.

Rules aligned with the query:
- Rotations/reflections allowed
- Pieces form one 8-connected assembly (side or vertex contact)
- Enclosed area = empty cells with no orthogonal path to the exterior
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

from ortools.sat.python import cp_model

Cell = Tuple[int, int]
Shape = frozenset[Cell]

RAW = {
    "F": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    "N": [(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
}

PIECES = ["F", "L", "N", "T", "Z"]
NEIGH8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def normalize(cells: Sequence[Cell]) -> Shape:
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def orientations(name: str) -> List[Shape]:
    base = RAW[name]
    seen: Set[Shape] = set()
    out: List[Shape] = []
    for flip in (False, True):
        pts = [(-r, c) if flip else (r, c) for r, c in base]
        for _ in range(4):
            pts = [(-c, r) for r, c in pts]
            sh = normalize(pts)
            if sh not in seen:
                seen.add(sh)
                out.append(sh)
    return out


ORIENTS = {p: orientations(p) for p in PIECES}


def shape_bbox(shape: Shape) -> Tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def all_placements(name: str, H: int, W: int) -> List[Shape]:
    placements: List[Shape] = []
    for orient in ORIENTS[name]:
        h, w = shape_bbox(orient)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                placements.append(frozenset((r0 + r, c0 + c) for r, c in orient))
    return placements


def ascii_board(occ: Dict[Cell, str], H: int, W: int, interior: Set[Cell]) -> str:
    lines = []
    for r in range(H):
        row = []
        for c in range(W):
            if (r, c) in occ:
                row.append(occ[(r, c)])
            elif (r, c) in interior:
                row.append("·")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def solve(H: int, W: int, time_limit: float, strict_fence: bool = False, fix_f: bool = True):
    model = cp_model.CpModel()
    placements: Dict[str, List[Shape]] = {p: all_placements(p, H, W) for p in PIECES}

    if fix_f:
        # Pin F to a single orientation near the grid center (translation freedom
        # of the whole assembly is removed, matching "F in a fixed position").
        fo = ORIENTS["F"][0]
        h, w = shape_bbox(fo)
        r0 = (H - h) // 2
        c0 = (W - w) // 2
        fixed = frozenset((r0 + r, c0 + c) for r, c in fo)
        placements["F"] = [fixed]
        print(f"F fixed at offset {(r0, c0)}", flush=True)

    for p in PIECES:
        print(f"{p}: {len(ORIENTS[p])} orients, {len(placements[p])} placements", flush=True)

    x = {
        p: [model.NewBoolVar(f"x_{p}_{i}") for i in range(len(placements[p]))]
        for p in PIECES
    }
    for p in PIECES:
        model.AddExactlyOne(x[p])

    cells = [(r, c) for r in range(H) for c in range(W)]
    occ = {(r, c): model.NewBoolVar(f"o_{r}_{c}") for r, c in cells}

    cell_to_lits: Dict[Cell, List] = defaultdict(list)
    f_cover: Dict[Cell, List] = defaultdict(list)
    for p in PIECES:
        for i, sh in enumerate(placements[p]):
            for cell in sh:
                cell_to_lits[cell].append(x[p][i])
                if p == "F":
                    f_cover[cell].append(x[p][i])

    for cell, lits in cell_to_lits.items():
        model.Add(sum(lits) <= 1)
        model.AddMaxEquality(occ[cell], lits)
    for cell in cells:
        if cell not in cell_to_lits:
            model.Add(occ[cell] == 0)

    # 8-connectivity of occupied cells via reachability from F (layered)
    # reach[k][c] = cell c is occupied and reachable from F in <= k steps
    max_steps = 25  # at most 25 occupied cells
    # Layer 0: cells covered by F
    reach = [{(r, c): model.NewBoolVar(f"r0_{r}_{c}") for r, c in cells}]
    for r, c in cells:
        if f_cover.get((r, c)):
            model.AddMaxEquality(reach[0][(r, c)], f_cover[(r, c)])
        else:
            model.Add(reach[0][(r, c)] == 0)

    for k in range(1, max_steps):
        layer = {(r, c): model.NewBoolVar(f"r{k}_{r}_{c}") for r, c in cells}
        for r, c in cells:
            # candidates: previous reach at c or at any 8-neighbor
            preds = [reach[k - 1][(r, c)]]
            for dr, dc in NEIGH8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    preds.append(reach[k - 1][(nr, nc)])
            # layer[c] <=> occ[c] AND OR(preds)
            or_pred = model.NewBoolVar(f"op{k}_{r}_{c}")
            model.AddMaxEquality(or_pred, preds)
            # layer = occ AND or_pred
            model.AddBoolAnd([occ[(r, c)], or_pred]).OnlyEnforceIf(layer[(r, c)])
            model.AddBoolOr([occ[(r, c)].Not(), or_pred.Not(), layer[(r, c)]])
        reach.append(layer)

    # All occupied cells reachable
    final = reach[-1]
    for r, c in cells:
        model.AddImplication(occ[(r, c)], final[(r, c)])

    # Interior variables
    interior = {(r, c): model.NewBoolVar(f"i_{r}_{c}") for r, c in cells}
    for r, c in cells:
        if r == 0 or c == 0 or r == H - 1 or c == W - 1:
            model.Add(interior[(r, c)] == 0)
        model.AddImplication(interior[(r, c)], occ[(r, c)].Not())
        neighbors = NEIGH8 if strict_fence else NEIGH4
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                model.AddBoolOr(
                    [interior[(r, c)].Not(), occ[(nr, nc)], interior[(nr, nc)]]
                )
            else:
                model.Add(interior[(r, c)] == 0)

    model.Maximize(sum(interior[(r, c)] for r, c in cells))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4

    print(f"Solving {H}x{W} (strict_fence={strict_fence}) ...", flush=True)
    t0 = time.time()
    status = solver.Solve(model)
    dt = time.time() - t0
    print(f"status={solver.StatusName(status)} time={dt:.1f}s obj={solver.ObjectiveValue()}", flush=True)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    area = int(solver.ObjectiveValue())
    occ_map = {}
    for p in PIECES:
        for i, sh in enumerate(placements[p]):
            if solver.Value(x[p][i]):
                for cell in sh:
                    occ_map[cell] = p
                break
    interior_set = {(r, c) for r, c in cells if solver.Value(interior[(r, c)])}
    art = ascii_board(occ_map, H, W, interior_set)
    print(art)
    return {
        "area": area,
        "optimal": status == cp_model.OPTIMAL,
        "H": H,
        "W": W,
        "strict_fence": strict_fence,
        "art": art,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=11)
    ap.add_argument("--W", type=int, default=11)
    ap.add_argument("--time", type=float, default=180.0)
    ap.add_argument("--strict-fence", action="store_true")
    args = ap.parse_args()

    result = solve(args.H, args.W, args.time, args.strict_fence)
    if result:
        with open("/workspace/pentomino-fence/cpsat_result.json", "w") as f:
            json.dump(result, f, indent=2)
        with open("/workspace/pentomino-fence/best.txt", "w") as f:
            f.write(f"area={result['area']}\n")
            f.write(f"optimal={result['optimal']}\n")
            f.write(result["art"] + "\n")


if __name__ == "__main__":
    main()
