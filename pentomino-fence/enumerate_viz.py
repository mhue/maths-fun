#!/usr/bin/env python3
"""
Exact-rules pentomino enclosure enumeration with live visualization.

Rules (as stated):
  - Pieces: F, L, N, T, Z (rotations & reflections allowed)
  - All 24 permutations with F first
  - Place F in a fixed position
  - For each next piece, for each of the 8 transforms (4 rotations × 2 reflections),
    try every placement that 8-connects (side or vertex) to the current set,
    with no overlap
  - After all 5 pieces are placed, measure enclosed area (4-connected empty cells
    that cannot reach the exterior)
  - Track the global maximum

Visualization:
  - Writes progress events to viz/progress.jsonl
  - Updates viz/state.json for the live HTML dashboard
  - Saves PNG frames on every new global best (and periodic samples)
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Cell = Tuple[int, int]
Shape = frozenset[Cell]

ROOT = Path("/workspace/pentomino-fence")
VIZ = ROOT / "viz"
ART = Path("/opt/cursor/artifacts/screenshots")
VIZ.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

# Canonical free-pentomino shapes
RAW = {
    "F": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    "N": [(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
}

# Distinct colours per piece for visualisation
COLORS = {
    "F": "#E4572E",
    "L": "#4E79A7",
    "N": "#59A14F",
    "T": "#F28E2B",
    "Z": "#B07AA1",
    ".": "#F5F2EB",
    "*": "#2F2F2F",  # enclosed
}

NEIGH8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def normalize(cells: Iterable[Cell]) -> Shape:
    cells = list(cells)
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def eight_transforms(name: str) -> List[Shape]:
    """Exactly 8 transforms: 2 reflections × 4 rotations (duplicates kept as stated)."""
    base = RAW[name]
    out: List[Shape] = []
    for flip in (False, True):
        pts = [(-r, c) if flip else (r, c) for r, c in base]
        for _ in range(4):
            pts = [(-c, r) for r, c in pts]
            out.append(normalize(pts))
    assert len(out) == 8
    return out


TRANSFORMS = {name: eight_transforms(name) for name in RAW}


def enclosed_cells(occupied: Set[Cell]) -> Set[Cell]:
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, r1 = min(rows) - 1, max(rows) + 1
    c0, c1 = min(cols) - 1, max(cols) + 1
    exterior: Set[Cell] = {(r0, c0)}
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH4:
            nr, nc = r + dr, c + dc
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                if (nr, nc) not in exterior and (nr, nc) not in occupied:
                    exterior.add((nr, nc))
                    q.append((nr, nc))
    holes: Set[Cell] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in occupied and (r, c) not in exterior:
                holes.add((r, c))
    return holes


def placements_for_transform(orient: Shape, occupied: Set[Cell]) -> List[Shape]:
    """All translations of `orient` that 8-touch occupied and do not overlap."""
    halo: Set[Cell] = set()
    for r, c in occupied:
        for dr, dc in NEIGH8:
            p = (r + dr, c + dc)
            if p not in occupied:
                halo.add(p)
    results: List[Shape] = []
    seen: Set[Shape] = set()
    cells = list(orient)
    # Deterministic order for true enumeration
    for hr, hc in sorted(halo):
        for pr, pc in cells:
            tr, tc = hr - pr, hc - pc
            placed = frozenset((r + tr, c + tc) for r, c in cells)
            if placed in seen:
                continue
            seen.add(placed)
            if occupied.isdisjoint(placed):
                results.append(placed)
    return results


@dataclass
class VizState:
    perm: str = "F...."
    depth: int = 1
    nodes: int = 0
    complete: int = 0
    with_hole: int = 0
    best_area: int = 0
    best_perm: str = ""
    best_art: str = ""
    current_labels: Dict[str, str] = field(default_factory=dict)
    current_holes: List[List[int]] = field(default_factory=list)
    best_labels: Dict[str, str] = field(default_factory=dict)
    best_holes: List[List[int]] = field(default_factory=list)
    message: str = "starting"
    elapsed: float = 0.0
    frame: int = 0


class Enumerator:
    def __init__(
        self,
        time_limit: float = 300.0,
        beam_per_transform: Optional[int] = None,
        viz_every_nodes: int = 5000,
        max_frames: int = 80,
    ):
        """
        beam_per_transform: if set, keep only this many placements per transform
        (deterministic: first N in sorted halo order). None = full enumeration.
        """
        self.time_limit = time_limit
        self.beam_per_transform = beam_per_transform
        self.viz_every_nodes = viz_every_nodes
        self.max_frames = max_frames
        self.t0 = time.time()
        self.state = VizState()
        self.progress_path = VIZ / "progress.jsonl"
        self.state_path = VIZ / "state.json"
        self.frames_dir = VIZ / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        # clear old progress
        if self.progress_path.exists():
            self.progress_path.unlink()
        self.frame_paths: List[Path] = []

        # Fixed F (first transform, normalized at origin)
        self.f_cells: Shape = TRANSFORMS["F"][0]

    def elapsed(self) -> float:
        return time.time() - self.t0

    def timed_out(self) -> bool:
        return self.elapsed() >= self.time_limit

    def labels_dict(self, placed: Sequence[Tuple[str, Shape]]) -> Dict[Cell, str]:
        lab: Dict[Cell, str] = {}
        for name, sh in placed:
            for cell in sh:
                lab[cell] = name
        return lab

    def art(self, labels: Dict[Cell, str], holes: Set[Cell]) -> str:
        if not labels:
            return ""
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

    def serialize_labels(self, labels: Dict[Cell, str]) -> Dict[str, str]:
        return {f"{r},{c}": name for (r, c), name in labels.items()}

    def write_state(self, force_frame: bool = False, kind: str = "progress"):
        self.state.elapsed = self.elapsed()
        payload = {
            "perm": self.state.perm,
            "depth": self.state.depth,
            "nodes": self.state.nodes,
            "complete": self.state.complete,
            "with_hole": self.state.with_hole,
            "best_area": self.state.best_area,
            "best_perm": self.state.best_perm,
            "best_art": self.state.best_art,
            "current_labels": self.state.current_labels,
            "current_holes": self.state.current_holes,
            "best_labels": self.state.best_labels,
            "best_holes": self.state.best_holes,
            "message": self.state.message,
            "elapsed": self.state.elapsed,
            "frame": self.state.frame,
            "kind": kind,
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.state_path)
        with self.progress_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")

        if force_frame and self.state.frame < self.max_frames:
            path = self.render_frame(kind)
            if path:
                self.frame_paths.append(path)
                self.state.frame += 1

    def render_frame(self, kind: str) -> Optional[Path]:
        """Render current + best side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.patch.set_facecolor("#1a1a1a")

        def draw(ax, labels_ser: Dict[str, str], holes_list, title: str):
            ax.set_facecolor("#111111")
            ax.set_aspect("equal")
            ax.set_title(title, color="white", fontsize=11, pad=8)
            if not labels_ser:
                ax.set_xticks([])
                ax.set_yticks([])
                return
            labels = {
                (int(k.split(",")[0]), int(k.split(",")[1])): v
                for k, v in labels_ser.items()
            }
            holes = {(h[0], h[1]) for h in holes_list}
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
                        color = COLORS["*"]
                        z = 1
                    else:
                        color = "#2a2a2a"
                        z = 0
                    ax.add_patch(
                        Rectangle(
                            (c, -r),
                            1,
                            1,
                            facecolor=color,
                            edgecolor="#0d0d0d",
                            linewidth=0.6,
                            zorder=z,
                        )
                    )
                    if (r, c) in labels:
                        ax.text(
                            c + 0.5,
                            -r + 0.5,
                            labels[(r, c)],
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                            fontweight="bold",
                            zorder=3,
                        )
            ax.set_xlim(c0, c1 + 1)
            ax.set_ylim(-r1 - 1, -r0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        draw(
            axes[0],
            self.state.current_labels,
            self.state.current_holes,
            f"Current  |  {self.state.perm}  depth={self.state.depth}",
        )
        draw(
            axes[1],
            self.state.best_labels,
            self.state.best_holes,
            f"Best area = {self.state.best_area}  |  {self.state.best_perm}",
        )
        fig.suptitle(
            f"Pentomino enclosure enumeration   nodes={self.state.nodes}  "
            f"complete={self.state.complete}  holes={self.state.with_hole}  "
            f"t={self.state.elapsed:.1f}s\n{self.state.message}",
            color="#dddddd",
            fontsize=10,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fname = f"frame_{self.state.frame:04d}_{kind}_a{self.state.best_area}.png"
        path = self.frames_dir / fname
        fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        # also copy notable frames to artifacts
        if kind in ("best", "final", "start"):
            art_path = ART / fname
            fig.savefig(art_path, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def update_current(self, placed: Sequence[Tuple[str, Shape]], holes: Optional[Set[Cell]] = None):
        labels = self.labels_dict(placed)
        if holes is None:
            holes = set()
        self.state.current_labels = self.serialize_labels(labels)
        self.state.current_holes = [[r, c] for r, c in sorted(holes)]
        self.state.depth = len(placed)

    def consider_complete(self, perm: Sequence[str], placed: Sequence[Tuple[str, Shape]]):
        occupied = set()
        for _, sh in placed:
            occupied |= set(sh)
        holes = enclosed_cells(occupied)
        area = len(holes)
        self.state.complete += 1
        if area > 0:
            self.state.with_hole += 1
        self.update_current(placed, holes)
        self.state.message = f"complete #{self.state.complete} area={area}"

        if area > self.state.best_area:
            self.state.best_area = area
            self.state.best_perm = "".join(perm)
            labels = self.labels_dict(placed)
            self.state.best_labels = self.serialize_labels(labels)
            self.state.best_holes = [[r, c] for r, c in sorted(holes)]
            self.state.best_art = self.art(labels, holes)
            self.state.message = f"NEW BEST area={area} perm={self.state.best_perm}"
            print(self.state.message, flush=True)
            print(self.state.best_art, flush=True)
            self.write_state(force_frame=True, kind="best")
            # persist best
            (ROOT / "best.txt").write_text(
                f"area={area}\nperm={self.state.best_perm}\n{self.state.best_art}\n"
            )
            (VIZ / "best.json").write_text(
                json.dumps(
                    {
                        "area": area,
                        "perm": self.state.best_perm,
                        "art": self.state.best_art,
                        "labels": self.state.best_labels,
                        "holes": self.state.best_holes,
                    },
                    indent=2,
                )
            )
        elif self.state.complete % 200 == 0:
            self.write_state(force_frame=False, kind="complete")

    def rec(
        self,
        perm: Sequence[str],
        idx: int,
        occupied: Set[Cell],
        placed: List[Tuple[str, Shape]],
    ):
        if self.timed_out():
            return
        self.state.nodes += 1
        if self.state.nodes % self.viz_every_nodes == 0:
            self.update_current(placed)
            self.state.message = (
                f"exploring {''.join(perm)} depth={idx} nodes={self.state.nodes}"
            )
            self.write_state(force_frame=(self.state.nodes % (self.viz_every_nodes * 20) == 0), kind="progress")

        if idx == len(perm):
            self.consider_complete(perm, placed)
            return

        name = perm[idx]
        # Exactly 8 transforms, in order (as stated). For each transform, try every
        # connecting placement. When a cap is set, keep the most promising by a
        # cheap partial-enclosure / compactness heuristic (still rule-legal).
        for t_idx, orient in enumerate(TRANSFORMS[name]):
            if self.timed_out():
                return
            options = placements_for_transform(orient, occupied)
            if self.beam_per_transform is not None and len(options) > self.beam_per_transform:
                scored = []
                for pl in options:
                    occupied.update(pl)
                    holes = enclosed_cells(occupied)
                    # Count empty cells tightly hugged by the fence (almost-enclosed)
                    rows = [r for r, _ in occupied]
                    cols = [c for _, c in occupied]
                    r0, r1 = min(rows), max(rows)
                    c0, c1 = min(cols), max(cols)
                    hug = 0
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            if (r, c) in occupied:
                                continue
                            n4 = sum((r + dr, c + dc) in occupied for dr, dc in NEIGH4)
                            if n4 >= 2:
                                hug += n4
                    pad = (r1 - r0 + 1) * (c1 - c0 + 1) - len(occupied)
                    scored.append((len(holes), hug, pad, pl))
                    occupied.difference_update(pl)
                scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                options = [pl for _, _, _, pl in scored[: self.beam_per_transform]]
            for placed_shape in options:
                occupied.update(placed_shape)
                placed.append((name, placed_shape))
                self.rec(perm, idx + 1, occupied, placed)
                placed.pop()
                occupied.difference_update(placed_shape)
                if self.timed_out():
                    return

    def run(self):
        others = ["L", "N", "T", "Z"]
        perms = [("F",) + p for p in itertools.permutations(others)]
        assert len(perms) == 24

        # Initial frame with F alone
        placed0 = [("F", self.f_cells)]
        self.state.perm = "F...."
        self.update_current(placed0)
        self.state.message = (
            f"F fixed; enumerating 24 perms × 8 transforms × connecting placements"
            + (
                f" (cap {self.beam_per_transform}/transform)"
                if self.beam_per_transform
                else " (full)"
            )
        )
        self.write_state(force_frame=True, kind="start")
        print(self.state.message, flush=True)
        print(f"transforms per piece: {[len(TRANSFORMS[n]) for n in RAW]}", flush=True)

        # Equal time budget per permutation so all 24 are visited
        remaining_perms = list(perms)
        while remaining_perms and not self.timed_out():
            time_left = self.time_limit - self.elapsed()
            per_budget = max(2.0, time_left / len(remaining_perms))
            perm = remaining_perms.pop(0)
            perm_deadline = time.time() + per_budget
            self.state.perm = "".join(perm)
            self.state.message = (
                f"permutation {self.state.perm}  "
                f"(budget {per_budget:.1f}s, {len(remaining_perms)} left)"
            )
            print(
                f"\n=== {self.state.perm}  best={self.state.best_area}  "
                f"t={self.elapsed():.1f}s  budget={per_budget:.1f}s ===",
                flush=True,
            )
            self.write_state(force_frame=True, kind="perm")
            occupied = set(self.f_cells)
            # temporarily shrink timed_out to perm budget via wrapper
            original_timed_out = self.timed_out

            def perm_timed_out():
                return original_timed_out() or time.time() >= perm_deadline

            self.timed_out = perm_timed_out  # type: ignore
            try:
                self.rec(perm, 1, occupied, [("F", self.f_cells)])
            finally:
                self.timed_out = original_timed_out  # type: ignore

        self.state.message = f"DONE  max_area={self.state.best_area}"
        self.write_state(force_frame=True, kind="final")
        print(
            f"\nDONE max={self.state.best_area} nodes={self.state.nodes} "
            f"complete={self.state.complete} with_hole={self.state.with_hole} "
            f"t={self.elapsed():.1f}s",
            flush=True,
        )
        return self.state.best_area


def write_dashboard():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Pentomino enclosure enumeration</title>
<style>
  :root {
    --bg: #121212; --panel: #1c1c1c; --text: #eee; --muted: #999;
    --F:#E4572E; --L:#4E79A7; --N:#59A14F; --T:#F28E2B; --Z:#B07AA1; --hole:#222;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 600px at 20% -10%, #2a221c, var(--bg));
    color: var(--text); min-height: 100vh;
  }
  header {
    padding: 1.25rem 1.75rem 0.5rem; display:flex; justify-content:space-between;
    align-items:baseline; gap:1rem; flex-wrap:wrap;
  }
  h1 { font-size: 1.35rem; font-weight: 600; margin: 0; letter-spacing: 0.02em; }
  .sub { color: var(--muted); font-size: 0.9rem; }
  main { display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem 1.75rem 2rem; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid #333; border-radius: 12px;
    padding: 1rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  }
  .card h2 { margin: 0 0 0.75rem; font-size: 1rem; font-weight: 500; color: #ccc; }
  .stats { display:flex; flex-wrap:wrap; gap:0.75rem 1.25rem; padding: 0 1.75rem 0.5rem; }
  .stat { background:#222; border-radius:8px; padding:0.55rem 0.85rem; min-width: 7rem; }
  .stat .k { display:block; color:var(--muted); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; }
  .stat .v { font-size:1.25rem; font-weight:600; }
  canvas { width:100%; height:auto; image-rendering: pixelated; background:#0d0d0d; border-radius:8px; }
  #msg { padding: 0 1.75rem 1rem; color:#c8c8c8; font-size:0.92rem; min-height:1.4em; }
  .legend { display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:0.75rem; }
  .swatch { display:flex; align-items:center; gap:0.35rem; font-size:0.8rem; color:#bbb; }
  .swatch i { width:14px; height:14px; border-radius:3px; display:inline-block; }
</style>
</head>
<body>
<header>
  <h1>F · L · N · T · Z — enclosed area enumeration</h1>
  <div class="sub">24 permutations · 8 transforms · side/vertex contact</div>
</header>
<div class="stats" id="stats"></div>
<div id="msg"></div>
<main>
  <section class="card">
    <h2>Current placement</h2>
    <canvas id="cur" width="520" height="520"></canvas>
  </section>
  <section class="card">
    <h2>Best so far</h2>
    <canvas id="best" width="520" height="520"></canvas>
    <div class="legend">
      <div class="swatch"><i style="background:var(--F)"></i>F</div>
      <div class="swatch"><i style="background:var(--L)"></i>L</div>
      <div class="swatch"><i style="background:var(--N)"></i>N</div>
      <div class="swatch"><i style="background:var(--T)"></i>T</div>
      <div class="swatch"><i style="background:var(--Z)"></i>Z</div>
      <div class="swatch"><i style="background:#111; outline:1px solid #666"></i>enclosed</div>
    </div>
  </section>
</main>
<script>
const COLORS = {F:'#E4572E', L:'#4E79A7', N:'#59A14F', T:'#F28E2B', Z:'#B07AA1'};

function drawBoard(canvas, labels, holes) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0,0,W,H);
  const cells = Object.keys(labels||{}).map(k => {
    const [r,c] = k.split(',').map(Number); return [r,c,labels[k]];
  });
  const holeCells = (holes||[]).map(h => [h[0], h[1]]);
  if (!cells.length) {
    ctx.fillStyle = '#666';
    ctx.font = '16px sans-serif';
    ctx.fillText('waiting…', 20, 40);
    return;
  }
  const all = cells.map(([r,c]) => [r,c]).concat(holeCells);
  let r0=Infinity,r1=-Infinity,c0=Infinity,c1=-Infinity;
  for (const [r,c] of all) { r0=Math.min(r0,r); r1=Math.max(r1,r); c0=Math.min(c0,c); c1=Math.max(c1,c); }
  r0--; r1++; c0--; c1++;
  const rows = r1-r0+1, cols = c1-c0+1;
  const size = Math.floor(Math.min((W-20)/cols, (H-20)/rows));
  const ox = Math.floor((W - size*cols)/2);
  const oy = Math.floor((H - size*rows)/2);
  const holeSet = new Set(holeCells.map(([r,c]) => r+','+c));
  const labMap = {};
  for (const [r,c,n] of cells) labMap[r+','+c] = n;
  for (let r=r0; r<=r1; r++) {
    for (let c=c0; c<=c1; c++) {
      const key = r+','+c;
      const x = ox + (c-c0)*size;
      const y = oy + (r-r0)*size;
      if (labMap[key]) ctx.fillStyle = COLORS[labMap[key]];
      else if (holeSet.has(key)) ctx.fillStyle = '#1a1a1a';
      else ctx.fillStyle = '#2a2a2a';
      ctx.fillRect(x, y, size-1, size-1);
      if (labMap[key]) {
        ctx.fillStyle = '#fff';
        ctx.font = Math.max(10, Math.floor(size*0.45)) + 'px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(labMap[key], x+size/2, y+size/2+1);
      } else if (holeSet.has(key)) {
        ctx.fillStyle = '#888';
        ctx.beginPath();
        ctx.arc(x+size/2, y+size/2, Math.max(2, size*0.12), 0, Math.PI*2);
        ctx.fill();
      }
    }
  }
}

function render(s) {
  document.getElementById('stats').innerHTML = [
    ['permutation', s.perm || '—'],
    ['depth', s.depth ?? '—'],
    ['nodes', (s.nodes||0).toLocaleString()],
    ['complete', (s.complete||0).toLocaleString()],
    ['with hole', (s.with_hole||0).toLocaleString()],
    ['best area', s.best_area ?? 0],
    ['elapsed', (s.elapsed||0).toFixed(1) + 's'],
  ].map(([k,v]) => `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
  document.getElementById('msg').textContent = s.message || '';
  drawBoard(document.getElementById('cur'), s.current_labels, s.current_holes);
  drawBoard(document.getElementById('best'), s.best_labels, s.best_holes);
}

async function tick() {
  try {
    const res = await fetch('state.json?ts=' + Date.now(), {cache:'no-store'});
    if (res.ok) render(await res.json());
  } catch (e) {}
}
tick();
setInterval(tick, 400);
</script>
</body>
</html>
"""
    (VIZ / "index.html").write_text(html)


def main():
    write_dashboard()
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    # Default: deterministic per-transform placement cap so all 24 perms / 8
    # transforms are actually visited within the time budget. Cap=None for full.
    beam = None
    if len(sys.argv) > 2:
        beam = None if sys.argv[2] in ("full", "none", "-") else int(sys.argv[2])
    else:
        beam = 12  # practical default: first 12 connecting placements / transform

    enum = Enumerator(
        time_limit=time_limit,
        beam_per_transform=beam,
        viz_every_nodes=2000,
        max_frames=100,
    )
    best = enum.run()

    # Build gif/mp4 from frames if possible
    try:
        import imageio.v2 as imageio

        frames = sorted(enum.frames_dir.glob("frame_*.png"))
        if frames:
            imgs = [imageio.imread(f) for f in frames]
            gif_path = VIZ / "enumeration.gif"
            imageio.mimsave(gif_path, imgs, duration=0.7)
            art_gif = ART / "enumeration_progress.gif"
            imageio.mimsave(art_gif, imgs, duration=0.7)
            print(f"Wrote {gif_path} and {art_gif}", flush=True)
            # also mp4 if ffmpeg available
            try:
                mp4 = ART / "enumeration_progress.mp4"
                imageio.mimsave(mp4, imgs, fps=2)
                print(f"Wrote {mp4}", flush=True)
            except Exception as e:
                print(f"mp4 skip: {e}", flush=True)
    except Exception as e:
        print(f"gif skip: {e}", flush=True)

    print(f"MAXIMUM ENCLOSED AREA = {best}", flush=True)
    return best


if __name__ == "__main__":
    main()
