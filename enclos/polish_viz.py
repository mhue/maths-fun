#!/usr/bin/env python3
"""Phase-2 polish under the same placement rules; streams into the live viz."""

from __future__ import annotations

import itertools
import json
import random
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from enumerate_viz import (
    ART,
    COLORS,
    NEIGH4,
    NEIGH8,
    TRANSFORMS,
    VIZ,
    enclosed_cells,
    placements_for_transform,
)


def art(labels, holes):
    cells = list(labels) + list(holes)
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    lines = []
    for r in range(r0, r1 + 1):
        row = []
        for c in range(c0, c1 + 1):
            if (r, c) in labels:
                row.append(labels[(r, c)])
            elif (r, c) in holes:
                row.append("·")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def is_conn(occ):
    s = next(iter(occ))
    seen = {s}
    q = deque([s])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH8:
            p = (r + dr, c + dc)
            if p in occ and p not in seen:
                seen.add(p)
                q.append(p)
    return len(seen) == len(occ)


def write_viz(holder, labels, holes, msg):
    area = holder["area"]
    ser = {f"{r},{c}": n for (r, c), n in labels.items()}
    holes_l = [[r, c] for r, c in sorted(holes)]
    payload = {
        "perm": "polish",
        "depth": 5,
        "nodes": holder["nodes"],
        "complete": holder["nodes"],
        "with_hole": holder["nodes"],
        "best_area": area,
        "best_perm": holder.get("perm", "polish"),
        "best_art": art(labels, holes),
        "current_labels": ser,
        "current_holes": holes_l,
        "best_labels": ser,
        "best_holes": holes_l,
        "message": msg,
        "elapsed": time.time() - holder["t0"],
        "frame": holder["frame"],
        "kind": "best",
    }
    (VIZ / "state.json").write_text(json.dumps(payload))
    (VIZ / "best.json").write_text(
        json.dumps(
            {
                "area": area,
                "perm": holder.get("perm", "polish"),
                "art": payload["best_art"],
                "labels": ser,
                "holes": holes_l,
            },
            indent=2,
        )
    )
    Path("/workspace/enclos/best.txt").write_text(
        f"area={area}\nperm={holder.get('perm','polish')}\n{payload['best_art']}\n"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#111111")
    cells = list(labels) + list(holes)
    r0 = min(r for r, _ in cells) - 1
    r1 = max(r for r, _ in cells) + 1
    c0 = min(c for _, c in cells) - 1
    c1 = max(c for _, c in cells) + 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in labels:
                color = COLORS[labels[(r, c)]]
                z = 2
            elif (r, c) in holes:
                color = "#222222"
                z = 1
            else:
                color = "#2a2a2a"
                z = 0
            ax.add_patch(
                Rectangle(
                    (c, -r), 1, 1, facecolor=color, edgecolor="#0d0d0d", lw=0.6, zorder=z
                )
            )
            if (r, c) in labels:
                ax.text(
                    c + 0.5,
                    -r + 0.5,
                    labels[(r, c)],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                    zorder=3,
                )
    ax.set_xlim(c0, c1 + 1)
    ax.set_ylim(-r1 - 1, -r0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Best area = {area}", color="white")
    fname = f"frame_polish_{holder['frame']:03d}_a{area}.png"
    fig.savefig(VIZ / "frames" / fname, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(ART / f"best_area_{area}.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    holder["frame"] += 1
    print(msg, flush=True)
    print(payload["best_art"], flush=True)


def hug_score(occ):
    rows = [r for r, _ in occ]
    cols = [c for _, c in occ]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    hug = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in occ:
                continue
            n4 = sum((r + dr, c + dc) in occ for dr, dc in NEIGH4)
            if n4 >= 2:
                hug += n4
    return hug


def main():
    data = json.loads((VIZ / "best.json").read_text())
    labels = {
        (int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in data["labels"].items()
    }
    holes = enclosed_cells(set(labels))
    holder = {
        "area": data["area"],
        "perm": data.get("perm", "FNLTZ"),
        "t0": time.time(),
        "frame": 0,
        "nodes": 0,
    }
    write_viz(holder, labels, holes, f"polish start area={holder['area']}")
    best_lab = labels
    rng = random.Random(1)
    deadline = time.time() + 160

    while time.time() < deadline:
        k = rng.choice([2, 2, 3])
        move = list(rng.sample(["L", "N", "T", "Z"], k))
        rest = {c: n for c, n in best_lab.items() if n not in move}
        occ0 = set(rest)

        def rec(idx, occ, lab):
            nonlocal best_lab
            holder["nodes"] += 1
            if idx == len(move):
                if not is_conn(occ):
                    return
                h = enclosed_cells(occ)
                a = len(h)
                if a > holder["area"]:
                    holder["area"] = a
                    holder["perm"] = "polish-" + "".join(move)
                    best_lab = dict(lab)
                    write_viz(
                        holder,
                        best_lab,
                        h,
                        f"NEW BEST area={a} moved={move}",
                    )
                return
            name = move[idx]
            opts = []
            for orient in TRANSFORMS[name]:
                opts.extend(placements_for_transform(orient, occ))
            if len(opts) > 90:
                scored = []
                for pl in opts:
                    occ2 = occ | pl
                    scored.append((len(enclosed_cells(occ2)), hug_score(occ2), pl))
                scored.sort(reverse=True)
                opts = [pl for _, _, pl in scored[:60]]
            for pl in opts:
                lab2 = dict(lab)
                for c in pl:
                    lab2[c] = name
                rec(idx + 1, occ | pl, lab2)

        rec(0, occ0, rest)

    # Constructive greedy pass over all 24 perms with full transform set
    print("constructive pass…", flush=True)
    fcells = frozenset(c for c, n in best_lab.items() if n == "F")
    for perm in [("F",) + p for p in itertools.permutations(["L", "N", "T", "Z"])]:
        if time.time() > holder["t0"] + 220:
            break
        occ = set(fcells)
        lab = {c: "F" for c in fcells}
        ok = True
        for name in perm[1:]:
            opts = []
            for orient in TRANSFORMS[name]:
                opts.extend(placements_for_transform(orient, occ))
            if not opts:
                ok = False
                break
            scored = [
                (len(enclosed_cells(occ | pl)), hug_score(occ | pl), pl) for pl in opts
            ]
            scored.sort(reverse=True)
            # try top few completions for last piece later; greedy here
            pl = scored[0][2]
            # for last two pieces, try top candidates more carefully
            if name == perm[-2] or name == perm[-1]:
                # keep exploring top ranked
                pass
            occ |= pl
            for c in pl:
                lab[c] = name
        if not ok:
            continue
        # re-place last two exhaustively
        p4, p5 = perm[-2], perm[-1]
        base_lab = {c: n for c, n in lab.items() if n not in (p4, p5)}
        base_occ = set(base_lab)
        opts4 = []
        for orient in TRANSFORMS[p4]:
            opts4.extend(placements_for_transform(orient, base_occ))
        local_best = holder["area"]
        local_lab = None
        local_holes = None
        for pl4 in opts4:
            occ2 = base_occ | pl4
            opts5 = []
            for orient in TRANSFORMS[p5]:
                opts5.extend(placements_for_transform(orient, occ2))
            for pl5 in opts5:
                occ3 = occ2 | pl5
                if not is_conn(occ3):
                    continue
                h = enclosed_cells(occ3)
                a = len(h)
                if a > local_best:
                    local_best = a
                    ll = dict(base_lab)
                    for c in pl4:
                        ll[c] = p4
                    for c in pl5:
                        ll[c] = p5
                    local_lab, local_holes = ll, h
        if local_lab is not None and local_best > holder["area"]:
            holder["area"] = local_best
            holder["perm"] = "".join(perm)
            best_lab = local_lab
            write_viz(
                holder,
                best_lab,
                local_holes,
                f"NEW BEST area={local_best} perm={''.join(perm)}",
            )

    print("FINAL", holder["area"], flush=True)


if __name__ == "__main__":
    main()
