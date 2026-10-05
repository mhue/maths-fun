#!/usr/bin/env python3
"""
Lean CP-SAT model: F fixed, maximize 4-connected enclosed area.
Connectivity of pieces enforced via 5-node cut constraints with
lazy-ish compact And encoding using only placement-index channeling.
"""

from __future__ import annotations

import argparse
import itertools
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
    out: List[Shape] = []
    for orient in ORIENTS[name]:
        h, w = shape_bbox(orient)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                out.append(frozenset((r0 + r, c0 + c) for r, c in orient))
    return out


def touches8(a: Shape, b: Shape) -> bool:
    bs = set(b)
    for r, c in a:
        for dr, dc in NEIGH8:
            if (r + dr, c + dc) in bs:
                return True
    return False


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


def solve(H: int, W: int, time_limit: float, require_connected: bool = True):
    model = cp_model.CpModel()

    fo = ORIENTS["F"][0]
    h, w = shape_bbox(fo)
    fr, fc = (H - h) // 2, (W - w) // 2
    f_fixed = frozenset((fr + r, fc + c) for r, c in fo)
    placements: Dict[str, List[Shape]] = {"F": [f_fixed]}
    for p in PIECES:
        if p != "F":
            placements[p] = all_placements(p, H, W)
        print(f"{p}: {len(placements[p])} placements", flush=True)

    x = {
        p: [model.NewBoolVar(f"x_{p}_{i}") for i in range(len(placements[p]))]
        for p in PIECES
    }
    for p in PIECES:
        model.AddExactlyOne(x[p])

    cells = [(r, c) for r in range(H) for c in range(W)]
    occ = {(r, c): model.NewBoolVar(f"o_{r}_{c}") for r, c in cells}
    cell_lits: Dict[Cell, List] = defaultdict(list)
    for p in PIECES:
        for i, sh in enumerate(placements[p]):
            for cell in sh:
                cell_lits[cell].append(x[p][i])
    for cell, lits in cell_lits.items():
        model.Add(sum(lits) <= 1)
        model.AddMaxEquality(occ[cell], lits)
    for cell in cells:
        if cell not in cell_lits:
            model.Add(occ[cell] == 0)

    if require_connected:
        # Compact connectivity: for each piece pair, AllowedAssignments on (id_a, id_b)
        # for touching pairs, then cut constraints via boolean edges.
        ids = {
            p: model.NewIntVar(0, len(placements[p]) - 1, f"id_{p}") for p in PIECES
        }
        for p in PIECES:
            model.Add(ids[p] == sum(i * x[p][i] for i in range(len(placements[p]))))

        edge = {}
        for a, b in itertools.combinations(PIECES, 2):
            touching = []
            pa, pb = placements[a], placements[b]
            for i, sa in enumerate(pa):
                for j, sb in enumerate(pb):
                    if touches8(sa, sb):
                        touching.append((i, j))
            print(f"touch {a}-{b}: {len(touching)}", flush=True)
            e = model.NewBoolVar(f"e_{a}_{b}")
            edge[(a, b)] = e
            # e => (ida,idb) in touching; and if (ida,idb) in touching we may set e
            # Encode: e == 1 iff chosen pair touches, via:
            # For each touching (i,j): create => e can be true
            # Actually we only need e_true if they touch for connectivity cuts:
            # e <= 1 if touch. Use: e => OR_{touching} (x[a][i] & x[b][j])
            # and we can force e == that OR.
            if not touching:
                model.Add(e == 0)
                continue
            pair_lits = []
            for i, j in touching:
                v = model.NewBoolVar(f"p_{a}{i}_{b}{j}")
                # v <=> x[a][i] & x[b][j]
                model.AddBoolAnd([x[a][i], x[b][j]]).OnlyEnforceIf(v)
                model.AddBoolOr([x[a][i].Not(), x[b][j].Not(), v])
                pair_lits.append(v)
            model.AddMaxEquality(e, pair_lits)

        # Partition cuts
        for mask in range(1, (1 << len(PIECES)) - 1):
            S = [PIECES[k] for k in range(len(PIECES)) if mask & (1 << k)]
            if PIECES[0] not in S:
                continue
            T = [p for p in PIECES if p not in S]
            cross = []
            for a in S:
                for b in T:
                    key = (a, b) if (a, b) in edge else (b, a)
                    cross.append(edge[key])
            model.AddBoolOr(cross)

    # Interior
    interior = {(r, c): model.NewBoolVar(f"i_{r}_{c}") for r, c in cells}
    for r, c in cells:
        if r in (0, H - 1) or c in (0, W - 1):
            model.Add(interior[(r, c)] == 0)
        model.AddImplication(interior[(r, c)], occ[(r, c)].Not())
        for dr, dc in NEIGH4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                model.AddBoolOr(
                    [interior[(r, c)].Not(), occ[(nr, nc)], interior[(nr, nc)]]
                )
            else:
                model.Add(interior[(r, c)] == 0)

    model.Maximize(sum(interior.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4
    print(f"Solving {H}x{W} ...", flush=True)
    t0 = time.time()
    status = solver.Solve(model)
    print(
        f"status={solver.StatusName(status)} time={time.time()-t0:.1f}s "
        f"obj={solver.ObjectiveValue()}",
        flush=True,
    )
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
    interior_set = {cell for cell in cells if solver.Value(interior[cell])}
    art = ascii_board(occ_map, H, W, interior_set)
    print(art)
    return {
        "area": area,
        "optimal": status == cp_model.OPTIMAL,
        "H": H,
        "W": W,
        "art": art,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--no-connect", action="store_true")
    args = ap.parse_args()
    result = solve(args.H, args.W, args.time, require_connected=not args.no_connect)
    if result:
        with open("/workspace/enclos/cpsat_result.json", "w") as f:
            json.dump(result, f, indent=2)
        with open("/workspace/enclos/best.txt", "w") as f:
            f.write(f"area={result['area']}\noptimal={result['optimal']}\n")
            f.write(result["art"] + "\n")


if __name__ == "__main__":
    main()
