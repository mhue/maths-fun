#!/usr/bin/env python3
"""
Beam / DFS search for max enclosed area with F,L,N,T,Z.

F is fixed in position and orientation (as specified).
All 24 permutations of the remaining pieces are explored.
Each piece may be rotated/reflected; must 8-connect to the current set.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
import time
from collections import deque
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

Cell = Tuple[int, int]
Shape = FrozenSet[Cell]

RAW = {
    "F": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    "N": [(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
}

NEIGH8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def normalize(cells) -> Shape:
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


ORIENTS = {n: orientations(n) for n in RAW}


def enclosed_area(occupied: Set[Cell]) -> int:
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, r1 = min(rows) - 1, max(rows) + 1
    c0, c1 = min(cols) - 1, max(cols) + 1
    exterior: Set[Cell] = set()
    q = deque([(r0, c0)])
    exterior.add((r0, c0))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH4:
            nr, nc = r + dr, c + dc
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                if (nr, nc) not in exterior and (nr, nc) not in occupied:
                    exterior.add((nr, nc))
                    q.append((nr, nc))
    area = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in occupied and (r, c) not in exterior:
                area += 1
    return area


def candidate_placements(orient: Shape, occupied: Set[Cell]) -> List[Shape]:
    halo: Set[Cell] = set()
    for r, c in occupied:
        for dr, dc in NEIGH8:
            p = (r + dr, c + dc)
            if p not in occupied:
                halo.add(p)
    results: List[Shape] = []
    seen: Set[Shape] = set()
    orient_list = list(orient)
    for hr, hc in halo:
        for pr, pc in orient_list:
            tr, tc = hr - pr, hc - pc
            placed = frozenset((r + tr, c + tc) for r, c in orient_list)
            if placed in seen:
                continue
            seen.add(placed)
            if occupied.isdisjoint(placed):
                results.append(placed)
    return results


def score_state(occupied: Set[Cell]) -> Tuple[int, int, int]:
    """Heuristic: prefer more enclosure, then compact perimeter potential."""
    area = enclosed_area(occupied)
    # Approximate "nearly enclosed" by counting empty cells with many occupied neighbors
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, r1 = min(rows) - 2, max(rows) + 2
    c0, c1 = min(cols) - 2, max(cols) + 2
    trapped = 0
    peri = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in occupied:
                continue
            n4 = sum((r + dr, c + dc) in occupied for dr, dc in NEIGH4)
            n8 = sum((r + dr, c + dc) in occupied for dr, dc in NEIGH8)
            if n4 >= 2:
                trapped += n4
            peri += n8
    return (area, trapped, peri)


def ascii_art(occupied: Set[Cell], labels: Dict[Cell, str]) -> str:
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    lines = []
    for r in range(r0, r1 + 1):
        line = []
        for c in range(c0, c1 + 1):
            line.append(labels.get((r, c), "."))
        lines.append("".join(line))
    return "\n".join(lines)


def beam_search_perm(perm: Sequence[str], beam_width: int = 2000):
    assert perm[0] == "F"
    f = set(ORIENTS["F"][0])
    # state: (occupied frozenset, labels dict as tuple items, list of shapes)
    beam = [(frozenset(f), (("F", frozenset(f)),))]

    for name in perm[1:]:
        candidates = []
        seen_occ: Set[FrozenSet[Cell]] = set()
        for occ_fs, placed in beam:
            occupied = set(occ_fs)
            for orient in ORIENTS[name]:
                for pl in candidate_placements(orient, occupied):
                    new_occ = occ_fs | pl
                    if new_occ in seen_occ:
                        continue
                    seen_occ.add(new_occ)
                    candidates.append((new_occ, placed + ((name, pl),)))
        # Rank by heuristic
        scored = []
        for occ_fs, placed in candidates:
            sc = score_state(set(occ_fs))
            scored.append((sc, occ_fs, placed))
        scored.sort(key=lambda t: t[0], reverse=True)
        beam = [(occ, placed) for _, occ, placed in scored[:beam_width]]

    best_area = -1
    best = None
    for occ_fs, placed in beam:
        area = enclosed_area(set(occ_fs))
        if area > best_area:
            best_area = area
            labels = {}
            for name, sh in placed:
                for cell in sh:
                    labels[cell] = name
            best = {
                "perm": list(perm),
                "area": area,
                "art": ascii_art(set(occ_fs), labels),
            }
    return best_area, best


def dfs_perm(perm: Sequence[str], deadline: float, best_holder: List):
    """Depth-first with pruning against best_holder[0]."""
    assert perm[0] == "F"
    f = set(ORIENTS["F"][0])
    nodes = [0]

    def rec(idx, occupied, placed):
        if time.time() > deadline:
            return
        nodes[0] += 1
        if idx == len(perm):
            area = enclosed_area(occupied)
            if area > best_holder[0]:
                best_holder[0] = area
                labels = {}
                for name, sh in placed:
                    for cell in sh:
                        labels[cell] = name
                best_holder[1] = {
                    "perm": list(perm),
                    "area": area,
                    "art": ascii_art(occupied, labels),
                }
                print(f"  NEW BEST {area}\n{best_holder[1]['art']}", flush=True)
            return

        name = perm[idx]
        # Generate and order candidates by heuristic (best first)
        cands = []
        for orient in ORIENTS[name]:
            for pl in candidate_placements(orient, occupied):
                occupied.update(pl)
                sc = score_state(occupied)
                occupied.difference_update(pl)
                cands.append((sc, pl))
        cands.sort(key=lambda t: t[0], reverse=True)

        # Limit branching for middle depths if needed
        limit = None
        if idx <= 2:
            limit = 80  # keep top placements early
        elif idx == 3:
            limit = 120
        if limit:
            cands = cands[:limit]

        for _, pl in cands:
            occupied.update(pl)
            placed.append((name, pl))
            rec(idx + 1, occupied, placed)
            placed.pop()
            occupied.difference_update(pl)
            if time.time() > deadline:
                return

    rec(1, f, [("F", frozenset(f))])
    return nodes[0]


def worker_beam(args):
    perm, bw = args
    t0 = time.time()
    area, wit = beam_search_perm(perm, bw)
    return "".join(perm), area, wit, time.time() - t0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "beam"
    others = ["L", "N", "T", "Z"]
    perms = [("F",) + p for p in itertools.permutations(others)]
    print({n: len(ORIENTS[n]) for n in ORIENTS}, flush=True)

    if mode == "beam":
        bw = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        print(f"Beam search width={bw} over {len(perms)} perms", flush=True)
        global_best = 0
        global_wit = None
        t0 = time.time()
        with mp.Pool(4) as pool:
            for perm_s, area, wit, dt in pool.imap_unordered(
                worker_beam, [(p, bw) for p in perms]
            ):
                print(f"{perm_s}: {area} ({dt:.1f}s)", flush=True)
                if area > global_best:
                    global_best = area
                    global_wit = wit
        print(f"\nMAX = {global_best} in {time.time()-t0:.1f}s")
        if global_wit:
            print(global_wit["art"])
            with open("/workspace/pentomino-fence/best.txt", "w") as f:
                f.write(f"area={global_best}\n")
                f.write(f"perm={''.join(global_wit['perm'])}\n")
                f.write(global_wit["art"] + "\n")
            with open("/workspace/pentomino-fence/beam_result.json", "w") as f:
                json.dump(global_wit, f, indent=2)

    elif mode == "dfs":
        # Run limited DFS on all perms with shared best
        seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        best_holder = [0, None]
        t0 = time.time()
        per_perm = seconds / len(perms)
        for perm in perms:
            deadline = time.time() + per_perm
            print(f"DFS {''.join(perm)} best={best_holder[0]}", flush=True)
            nodes = dfs_perm(perm, deadline, best_holder)
            print(f"  nodes~{nodes} best={best_holder[0]}", flush=True)
        print(f"\nMAX = {best_holder[0]} in {time.time()-t0:.1f}s")
        if best_holder[1]:
            print(best_holder[1]["art"])
            with open("/workspace/pentomino-fence/best.txt", "w") as f:
                f.write(f"area={best_holder[0]}\n")
                f.write(best_holder[1]["art"] + "\n")


if __name__ == "__main__":
    main()
