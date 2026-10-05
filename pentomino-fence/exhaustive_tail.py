#!/usr/bin/env python3
"""
More aggressive search: enumerate diverse early placements, then
exhaustively optimize the last two pieces for enclosure.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

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


def all_touching(name: str, occupied: Set[Cell]) -> List[Shape]:
    halo = set()
    for r, c in occupied:
        for dr, dc in NEIGH8:
            p = (r + dr, c + dc)
            if p not in occupied:
                halo.add(p)
    out, seen = [], set()
    for orient in ORIENTS[name]:
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


def art(labels: Dict[Cell, str]) -> str:
    r0 = min(r for r, _ in labels)
    r1 = max(r for r, _ in labels)
    c0 = min(c for _, c in labels)
    c1 = max(c for _, c in labels)
    lines = []
    for r in range(r0, r1 + 1):
        lines.append("".join(labels.get((r, c), ".") for c in range(c0, c1 + 1)))
    return "\n".join(lines)


def best_completion(occupied: Set[Cell], labels_base: Dict[Cell, str], p4: str, p5: str):
    """Exhaustively place last two pieces; return best area and labels."""
    best_a = -1
    best_lab = None
    opts4 = all_touching(p4, occupied)
    for pl4 in opts4:
        occ2 = occupied | pl4
        opts5 = all_touching(p5, occ2)
        for pl5 in opts5:
            occ3 = occ2 | pl5
            a = enclosed_area(occ3)
            if a > best_a:
                best_a = a
                lab = dict(labels_base)
                for cell in pl4:
                    lab[cell] = p4
                for cell in pl5:
                    lab[cell] = p5
                best_lab = lab
    return best_a, best_lab


def main():
    rng = random.Random(0)
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    # How many random prefixes (3 pieces after F) to try per permutation
    prefixes_per_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    perms = [("F",) + p for p in itertools.permutations(["L", "N", "T", "Z"])]
    best = -1
    best_info = None
    t0 = time.time()
    total = 0

    print("start", flush=True)
    while time.time() - t0 < time_limit:
        for perm in perms:
            if time.time() - t0 >= time_limit:
                break
            # Build diverse prefixes of first 3 pieces (F + 2)
            p1, p2, p3, p4, p5 = perm
            assert p1 == "F"
            f = frozenset(ORIENTS["F"][0])
            occ0 = set(f)
            lab0 = {c: "F" for c in f}

            opts1 = all_touching(p2, occ0)
            # sample starts
            starts = opts1 if len(opts1) <= 30 else rng.sample(opts1, 30)
            for pl1 in starts:
                if time.time() - t0 >= time_limit:
                    break
                occ1 = occ0 | pl1
                lab1 = dict(lab0)
                for c in pl1:
                    lab1[c] = p2
                opts2 = all_touching(p3, occ1)
                picks2 = opts2 if len(opts2) <= 25 else rng.sample(opts2, 25)
                for pl2 in picks2:
                    occ2 = occ1 | pl2
                    lab2 = dict(lab1)
                    for c in pl2:
                        lab2[c] = p3
                    a, lab = best_completion(occ2, lab2, p4, p5)
                    total += 1
                    if a > best:
                        best = a
                        best_info = {"area": a, "perm": list(perm), "art": art(lab)}
                        print(f"BEST {best} perm={''.join(perm)} n={total} t={time.time()-t0:.1f}s", flush=True)
                        print(best_info["art"], flush=True)
            # also a few fully random deeper samples for this perm
            for _ in range(5):
                occ = set(f)
                lab = {c: "F" for c in f}
                ok = True
                for name in (p2, p3):
                    opts = all_touching(name, occ)
                    if not opts:
                        ok = False
                        break
                    pl = rng.choice(opts)
                    occ |= pl
                    for c in pl:
                        lab[c] = name
                if not ok:
                    continue
                a, labf = best_completion(occ, lab, p4, p5)
                total += 1
                if a > best:
                    best = a
                    best_info = {"area": a, "perm": list(perm), "art": art(labf)}
                    print(f"BEST {best} perm={''.join(perm)} n={total} t={time.time()-t0:.1f}s", flush=True)
                    print(best_info["art"], flush=True)

        print(f"pass done best={best} n={total} t={time.time()-t0:.1f}s", flush=True)

    print(f"FINAL {best}", flush=True)
    if best_info:
        with open("/workspace/pentomino-fence/best.txt", "w") as f:
            f.write(f"area={best}\nperm={''.join(best_info['perm'])}\n")
            f.write(best_info["art"] + "\n")
        with open("/workspace/pentomino-fence/search_result.json", "w") as f:
            json.dump(best_info, f, indent=2)


if __name__ == "__main__":
    main()
