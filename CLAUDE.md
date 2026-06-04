# CubeTidy — notes for working in this repo

Browser app: scan a Rubik's cube with the camera, solve it, overlay AR arrows for
each move. Vanilla JS + Canvas, bundled by Vite. No framework.

## Commands

- `npm run dev` — dev server on `localhost` (camera needs https or localhost).
- `npm run build` — production build; also the fastest way to catch import/syntax errors.
- `npm test` — end-to-end pipeline test (`test/pipeline.test.mjs`), no browser needed.

## Committing

- **Commit often**, in small logical chunks. Splitting one change across several
  commits is fine and encouraged.
- **Linear history only**: commit straight to `main`. No branches, no PRs, no
  pushing — local commits only.
- **Never commit when tests are known broken.** Run `npm test` first; if it fails,
  fix it (or don't commit that piece) before committing.

## Architecture (data flow)

`camera.js` → `detection.js` (sample grid) → `colors.js` (classify) →
`cube-state.js` (assemble + validate) → `solver.js` → `overlay.js` (draw arrows),
orchestrated by `main.js` (state machine + rAF render loop).

The single visible surface is `<canvas>`; the `<video>` is hidden and drawn into
the canvas each frame, so all sampling and overlays share one coordinate space.

## Things that are easy to get wrong

- **Facelet string is Kociemba `URFDLB` order**, each face row-major, characters
  are face letters (not colors). Centers sit at string indices 4,13,22,31,40,49.
  `cubejs` requires `Cube.initSolver()` once (slow table build) before `solve()`.

- **Scan order is `F, R, B, L, U, D`** (`SCAN_STEPS` in `cube-state.js`). The
  physical holding instruction for each face is chosen so the camera's row-major
  3×3 maps *directly* onto that face's facelet positions — turn the cube *left*
  for the four sides (keeping the top fixed), then tilt top/bottom toward the
  camera for U/D. If you change an instruction, you change the mapping; re-derive
  carefully or the solve will be wrong while everything still "looks" fine.

- **Arrow directions** in `overlay.js` (`ARROWS`) are derived for the fixed frame
  "front center toward camera, top center up": `U`→top row LEFT, `D`→bottom row
  RIGHT, `R`→right col UP, `L`→left col DOWN, `F`→clockwise, `B`→counter-clockwise
  (as seen from the front). `'` inverts; `2` is the same direction twice. Centers
  never move under face turns, so these stay valid for the whole solve as long as
  the user keeps that orientation.

- **Color classification is calibration-based**, not fixed thresholds: the six
  captured centers are the references, every sticker is matched to the nearest in
  CIE-Lab. This guarantees centers map to themselves and survives lighting shifts.
  Red↔orange is the residual weak spot under warm light.

- **Mirroring is display-only** — a `scaleX(-1)` CSS class (`.mirrored`) on the
  `<canvas>`. CSS transforms don't touch the canvas backing store, so
  `getImageData` sampling (and the solve) keep reading the true frame. Don't
  "fix" this by flipping the video into the canvas buffer — that would mirror the
  sampled cells and produce a wrong cube state. Arrows stay correct because they
  are glued to the cube and flip with it; canvas text is avoided for the same
  reason (it would render reversed), so move labels live in the HTML status bar.

- `cubejs` is CommonJS; `vite.config.js` pre-bundles it (`optimizeDeps.include`).
  Keep that if you touch the Vite config.

## Scope / next steps

This is the reliable "guided scan → solve → on-face arrows" design, **not**
continuous 3D tracking of a freely moving cube. `detection.js` is the intended
seam for that upgrade (contour detection + 6-DoF pose via `solvePnP`, 3D arrows
with three.js). Classification could move to k-means over all 54 samples to
better separate red/orange. The pixel-sampling layer is the only part `npm test`
can't cover — verify camera changes in a real browser.
