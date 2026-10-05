#!/usr/bin/env python3
"""Fast constructive + local-search maximizer for F,L,N,T,Z enclosure."""

from __future__ import annotations

import itertools
import json
import random
import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

Cell = Tuple[int, int]
Shape = FrozenSet[Cell]

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


def normalize(cells) -> Shape:
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def orientations(name: str) -> List[Shape]:
    base = RAW[name]
    seen, out = set(), []
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
    exterior = {(r0, c0)}
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH4:
            nr, nc = r + dr, c + dc
            if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in exterior and (nr, nc) not in occupied:
                exterior.add((nr, nc))
                q.append((nr, nc))
    return (r1 - r0 + 1) * (c1 - c0 + 1) - len(occupied) - len(exterior)


def placements_touching(orient: Shape, occupied: Set[Cell]) -> List[Shape]:
    halo = set()
    for r, c in occupied:
        for dr, dc in NEIGH8:
            p = (r + dr, c + dc)
            if p not in occupied:
                halo.add(p)
    out, seen = [], set()
    ol = list(orient)
    for hr, hc in halo:
        for pr, pc in ol:
            tr, tc = hr - pr, hc - pc
            placed = frozenset((r + tr, c + tc) for r, c in ol)
            if placed in seen:
                continue
            seen.add(placed)
            if occupied.isdisjoint(placed):
                out.append(placed)
    return out


def all_placements_for_piece(name: str, occupied: Set[Cell]) -> List[Shape]:
    res = []
    for orient in ORIENTS[name]:
        res.extend(placements_touching(orient, occupied))
    return res


def ascii_art(labels: Dict[Cell, str]) -> str:
    cells = list(labels)
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    # also show enclosed dots
    occ = set(cells)
    rows = []
    for r in range(r0, r1 + 1):
        row = []
        for c in range(c0, c1 + 1):
            if (r, c) in labels:
                row.append(labels[(r, c)])
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows)


def random_assembly(perm: List[str], rng: random.Random) -> Optional[Dict[str, Shape]]:
    """Place pieces in order, choosing random valid placement each time."""
    placed: Dict[str, Shape] = {"F": frozenset(ORIENTS["F"][0])}
    occupied = set(placed["F"])
    for name in perm[1:]:
        opts = all_placements_for_piece(name, occupied)
        if not opts:
            return None
        choice = rng.choice(opts)
        placed[name] = choice
        occupied.update(choice)
    return placed


def greedy_assembly(perm: List[str], rng: random.Random, samples: int = 40) -> Optional[Dict[str, Shape]]:
    """At each step, sample placements and pick the one maximizing heuristic."""
    placed: Dict[str, Shape] = {"F": frozenset(ORIENTS["F"][0])}
    occupied = set(placed["F"])
    for name in perm[1:]:
        opts = all_placements_for_piece(name, occupied)
        if not opts:
            return None
        if len(opts) > samples:
            opts = rng.sample(opts, samples)
        best_pl, best_sc = None, (-1, -1)
        for pl in opts:
            occupied.update(pl)
            area = enclosed_area(occupied)
            # favor larger bbox holes potential: more occupied neighbors on empty
            sc = (area, -len(occupied))  # primarily area
            # secondary: count empty cells with >=3 occupied 4-neighbors
            rows = [r for r, _ in occupied]
            cols = [c for _, c in occupied]
            bonus = 0
            for r in range(min(rows), max(rows) + 1):
                for c in range(min(cols), max(cols) + 1):
                    if (r, c) in occupied:
                        continue
                    if sum((r + dr, c + dc) in occupied for dr, dc in NEIGH4) >= 3:
                        bonus += 1
            sc = (area, bonus)
            occupied.difference_update(pl)
            if sc > best_sc:
                best_sc, best_pl = sc, pl
        placed[name] = best_pl
        occupied.update(best_pl)
    return placed


def mutate(placed: Dict[str, Shape], perm: List[str], rng: random.Random) -> Optional[Dict[str, Shape]]:
    """Remove a non-F suffix piece and try to re-place remaining in order."""
    # Re-place from a random cut index
    cut = rng.randint(1, len(perm) - 1)
    new_placed = {"F": placed["F"]}
    occupied = set(placed["F"])
    # Keep pieces before cut if we rebuild in perm order from scratch using
    # old positions when possible — simpler: rebuild from cut with random.
    for name in perm[1:cut]:
        new_placed[name] = placed[name]
        occupied.update(placed[name])
    for name in perm[cut:]:
        opts = all_placements_for_piece(name, occupied)
        if not opts:
            return None
        # Prefer placements that increase enclosure
        sample = opts if len(opts) <= 60 else rng.sample(opts, 60)
        best_pl, best_sc = None, -1
        for pl in sample:
            occupied.update(pl)
            sc = enclosed_area(occupied)
            occupied.difference_update(pl)
            if sc > best_sc:
                best_sc, best_pl = sc, pl
        # epsilon random
        if rng.random() < 0.15:
            best_pl = rng.choice(sample)
        new_placed[name] = best_pl
        occupied.update(best_pl)
    return new_placed


def evaluate(placed: Dict[str, Shape]) -> Tuple[int, Set[Cell], Dict[Cell, str]]:
    labels = {}
    for name, sh in placed.items():
        for cell in sh:
            labels[cell] = name
    occ = set(labels)
    return enclosed_area(occ), occ, labels


def main():
    import sys

    rng = random.Random(42)
    others = ["L", "N", "T", "Z"]
    perms = [["F"] + list(p) for p in itertools.permutations(others)]

    best_area = -1
    best_info = None
    t0 = time.time()
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    trials = 0

    print("orients:", {k: len(v) for k, v in ORIENTS.items()}, flush=True)

    while time.time() - t0 < time_limit:
        perm = rng.choice(perms)
        # constructive
        if rng.random() < 0.5:
            placed = greedy_assembly(perm, rng, samples=50)
        else:
            placed = random_assembly(perm, rng)
        if placed is None:
            continue
        area, _, labels = evaluate(placed)
        trials += 1

        # local search
        cur = placed
        cur_area = area
        for _ in range(30):
            nxt = mutate(cur, perm, rng)
            if nxt is None:
                continue
            a, _, lab = evaluate(nxt)
            if a >= cur_area or rng.random() < 0.05:
                cur, cur_area, labels = nxt, a, lab
            if cur_area > best_area:
                best_area = cur_area
                best_info = {
                    "area": best_area,
                    "perm": perm,
                    "art": ascii_art(labels),
                }
                print(f"BEST {best_area} perm={''.join(perm)} trials={trials} t={time.time()-t0:.1f}s", flush=True)
                print(best_info["art"], flush=True)

        if cur_area > best_area:
            best_area = cur_area
            best_info = {
                "area": best_area,
                "perm": perm,
                "art": ascii_art(labels),
            }
            print(f"BEST {best_area} perm={''.join(perm)} trials={trials} t={time.time()-t0:.1f}s", flush=True)
            print(best_info["art"], flush=True)

        if trials % 200 == 0:
            print(f"... trials={trials} best={best_area} t={time.time()-t0:.1f}s", flush=True)

    print(f"\nDONE best={best_area} trials={trials}", flush=True)
    if best_info:
        with open("/workspace/pentomino-fence/best.txt", "w") as f:
            f.write(f"area={best_area}\n")
            f.write(f"perm={''.join(best_info['perm'])}\n")
            f.write(best_info["art"] + "\n")
        with open("/workspace/pentomino-fence/search_result.json", "w") as f:
            json.dump(best_info, f, indent=2)


if __name__ == "__main__":
    import sys
    main()
