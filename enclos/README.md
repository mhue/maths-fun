# enclos — pentomino enclosed-area enumeration (F, L, N, T, Z)

## Rules

- Pieces: **F, L, N, T, Z** (rotations and reflections allowed — 8 transforms each).
- Enumerate all **24 permutations** with **F first**.
- Place **F in a fixed position**.
- For each next piece, for each of the 8 transforms, try placements that
  **8-connect** to the already-placed set (share a side **or** a vertex),
  with **no overlap**.
- After all five pieces are placed, measure the **enclosed area**: empty cells
  with no orthogonal (4-connected) path to the exterior.
- Report the **maximum** enclosed area found.

## Live visualization

```bash
python3 viz_server.py          # http://127.0.0.1:8765/
python3 enumerate_viz.py 360 5 # timed sweep of all 24 perms
python3 polish_viz.py          # deeper legal re-placement polish
```

The dashboard (`viz/index.html`) polls `viz/state.json` and shows the current
partial placement beside the best enclosure found so far.

## Result

**Maximum enclosed area found: 20**

Example (permutation growth order `F → L → N → T → Z`):

```
..NNNZZ..
.NN···Z..
F·····ZZ.
FFF····T.
.F·····T.
..L···TTT
..LLLL...
```

(· = enclosed cell)

## Notes on completeness

A fully exhaustive search of every connecting translation at every depth is
enormous (roughly \(O(100^4)\) leaves per permutation). The enumerator:

1. Visits all **24 permutations** and all **8 transforms** per piece.
2. Ranks connecting placements and explores the most promising ones under a
   time budget (see `enumerate_viz.py`).
3. Runs a polish phase that re-places 2–3 pieces using the same legal moves.

The configuration of area 20 is reachable by sequential 8-connected placement
starting from fixed F.
