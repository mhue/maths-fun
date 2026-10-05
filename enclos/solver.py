#!/usr/bin/env python3
"""
Maximum enclosed area by pentominoes F, L, N, T, Z.

Rules:
- Pieces may be rotated and reflected.
- All 24 permutations with F first.
- Place F fixed; each next piece must 8-connect (side or vertex) to the
  already-placed set, without overlapping.
- Enclosed area = number of empty cells in finite 4-connected components
  (cannot reach the exterior via orthogonal moves).
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import sys
import time
from collections import deque
from typing import FrozenSet, Iterable, List, Sequence, Set, Tuple

Cell = Tuple[int, int]
Shape = FrozenSet[Cell]


# Canonical shapes (as free pentominoes)
RAW = {
    "F": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    "N": [(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
}


def normalize(cells: Iterable[Cell]) -> Shape:
    cells = list(cells)
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def all_orientations(name: str) -> List[Shape]:
    base = RAW[name]
    seen: Set[Shape] = set()
    out: List[Shape] = []
    for flip in (False, True):
        pts = [(-r, c) if flip else (r, c) for r, c in base]
        for _ in range(4):
            pts = [(-c, r) for r, c in pts]  # 90° CCW
            sh = normalize(pts)
            if sh not in seen:
                seen.add(sh)
                out.append(sh)
    return out


ORIENTATIONS = {name: all_orientations(name) for name in RAW}
for name, orients in ORIENTATIONS.items():
    print(f"{name}: {len(orients)} unique orientations", file=sys.stderr)


NEIGH8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def touches8(a: Shape, occupied: Set[Cell]) -> bool:
    for r, c in a:
        for dr, dc in NEIGH8:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def candidate_placements(orient: Shape, occupied: Set[Cell]) -> List[Shape]:
    """All translations of orient that 8-touch occupied and do not overlap."""
    if not occupied:
        return [orient]

    # Halo: empty cells 8-adjacent to occupied
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
            # Map (pr,pc) -> (hr,hc)
            tr, tc = hr - pr, hc - pc
            placed = frozenset((r + tr, c + tc) for r, c in orient_list)
            if placed in seen:
                continue
            seen.add(placed)
            if placed.isdisjoint(occupied):
                # By construction one cell is in halo, so it 8-touches
                results.append(placed)
    return results


def enclosed_area(occupied: Set[Cell]) -> int:
    if not occupied:
        return 0
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, r1 = min(rows) - 1, max(rows) + 1
    c0, c1 = min(cols) - 1, max(cols) + 1

    # Flood-fill exterior from a corner of the padded bbox (guaranteed empty)
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

    # All empty cells inside bbox that are not exterior are enclosed
    area = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in occupied and (r, c) not in exterior:
                area += 1
    return area


def ascii_art(occupied: Set[Cell], labels: dict[Cell, str] | None = None) -> str:
    if not occupied:
        return ""
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    lines = []
    for r in range(r0, r1 + 1):
        line = []
        for c in range(c0, c1 + 1):
            if (r, c) in occupied:
                line.append(labels[(r, c)] if labels else "#")
            else:
                line.append(".")
        lines.append("".join(line))
    return "\n".join(lines)


def search_permutation(
    perm: Sequence[str],
    best_shared: mp.Value | None = None,
    find_one_at_least: int = 0,
) -> Tuple[int, dict | None]:
    """
    Exhaustive search for one permutation (F already first).
    Returns (best_area, witness_info or None).
    """
    assert perm[0] == "F"
    # Fix F at a canonical absolute position (already normalized orientation 0)
    f_orient = ORIENTATIONS["F"][0]
    # Place F centered-ish at origin (normalized so min is 0)
    f_cells = set(f_orient)

    best = 0
    witness = None
    nodes = [0]

    def rec(idx: int, occupied: Set[Cell], placed_list: List[Tuple[str, Shape]]):
        nonlocal best, witness
        nodes[0] += 1
        if idx == len(perm):
            area = enclosed_area(occupied)
            if area > best:
                best = area
                labels = {}
                for name, sh in placed_list:
                    for cell in sh:
                        labels[cell] = name
                witness = {
                    "perm": list(perm),
                    "area": area,
                    "art": ascii_art(occupied, labels),
                    "occupied": sorted(occupied),
                    "labels": {f"{r},{c}": lab for (r, c), lab in labels.items()},
                }
                if best_shared is not None:
                    with best_shared.get_lock():
                        if area > best_shared.value:
                            best_shared.value = area
            return

        # Optional pruning against global best (weak)
        name = perm[idx]
        for orient in ORIENTATIONS[name]:
            for placed in candidate_placements(orient, occupied):
                # quick reject? skip
                occupied.update(placed)
                placed_list.append((name, placed))
                rec(idx + 1, occupied, placed_list)
                placed_list.pop()
                occupied.difference_update(placed)
                if find_one_at_least and best >= find_one_at_least:
                    return

    rec(1, f_cells, [("F", frozenset(f_cells))])
    return best, witness, nodes[0]


def _worker(args):
    perm, shared_val = args
    t0 = time.time()
    best, witness, nodes = search_permutation(perm, best_shared=shared_val)
    dt = time.time() - t0
    return "".join(perm), best, witness, nodes, dt


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    others = ["L", "N", "T", "Z"]
    perms = [("F",) + p for p in itertools.permutations(others)]
    print(f"{len(perms)} permutations", flush=True)

    if mode == "probe":
        # Measure branching on first perm with depth limit stats
        perm = perms[0]
        occupied = set(ORIENTATIONS["F"][0])
        for depth, name in enumerate(perm[1:], start=1):
            total = 0
            for orient in ORIENTATIONS[name]:
                total += len(candidate_placements(orient, occupied))
            print(f"After {depth-1} pieces, candidates for {name}: {total}")
            # place first candidate of first orient to continue probe
            placed = candidate_placements(ORIENTATIONS[name][0], occupied)[0]
            occupied.update(placed)
            print(f"  enclosed so far: {enclosed_area(occupied)}")
        return

    if mode == "one":
        perm = perms[0]
        print(f"Searching {''.join(perm)} ...", flush=True)
        t0 = time.time()
        best, witness, nodes = search_permutation(perm)
        print(f"best={best} nodes={nodes} time={time.time()-t0:.1f}s")
        if witness:
            print(witness["art"])
        return

    # Full parallel search
    manager_best = mp.Value("i", 0)
    args = [(perm, manager_best) for perm in perms]
    global_best = 0
    global_witness = None
    t0 = time.time()
    with mp.Pool(processes=min(4, len(perms))) as pool:
        for perm_s, best, witness, nodes, dt in pool.imap_unordered(_worker, args):
            print(
                f"{perm_s}: best={best} nodes={nodes} time={dt:.1f}s "
                f"(global={manager_best.value})",
                flush=True,
            )
            if best > global_best and witness is not None:
                global_best = best
                global_witness = witness

    print(f"\nMAXIMUM ENCLOSED AREA = {global_best}")
    print(f"Total time: {time.time()-t0:.1f}s")
    if global_witness:
        print(f"Permutation: {''.join(global_witness['perm'])}")
        print(global_witness["art"])
        # Save witness
        with open("/workspace/enclos/best.txt", "w") as f:
            f.write(f"area={global_best}\n")
            f.write(f"perm={''.join(global_witness['perm'])}\n")
            f.write(global_witness["art"] + "\n")


if __name__ == "__main__":
    main()
