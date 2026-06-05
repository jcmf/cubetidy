# CubeTidy — notes for working in this repo

Browser app: scan a Rubik's cube with the camera, solve it, overlay AR arrows for
each move. Vanilla JS + Canvas, bundled by Vite. No framework.

## Commands

- `npm run dev` — dev server on `localhost` (camera needs https or localhost).
- `npm run build` — production build; also the fastest way to catch import/syntax errors.
- `npm test` — pipeline + scan-geometry + corner-geometry tests, no browser needed.
- `npm run preview:check` — render the overlay/glyph layer with skia-canvas (no
  browser) and pixel-diff against committed goldens in `tools/preview/golden/`.
  Run after touching `overlay.js`/`glyph.js`; a mismatch writes a red-highlighted
  `tools/preview/out/<scene>-diff.png` and exits non-zero. If the change is
  intentional, `npm run preview:update` and review the golden image diff. NOT part
  of `npm test` (golden diffs are platform-sensitive). Covers overlay *geometry*
  only — not the composited UI (no camera, CSS mirror, or HTML chrome).

## Committing

- **Commit often**, in small logical chunks. Splitting one change across several
  commits is fine and encouraged.
- **Linear history only**: commit straight to `main`. No branches, no PRs, no
  pushing — local commits only.
- **Never commit when tests are known broken.** Run `npm test` first; if it fails,
  fix it (or don't commit that piece) before committing.

## Architecture (data flow)

`camera.js` → `detection.js` (corner-on sampling) → `colors.js` (classify) →
`cube-state.js` (assemble + validate) → `solver.js` → `overlay.js` (draw arrows),
orchestrated by `main.js` (state machine + rAF render loop).

The single visible surface is `<canvas>`; the `<video>` is hidden and drawn into
the canvas each frame, so all sampling and overlays share one coordinate space.

## Things that are easy to get wrong

- **Facelet string is Kociemba `URFDLB` order**, each face row-major, characters
  are face letters (not colors). Centers sit at string indices 4,13,22,31,40,49.
  `cubejs` requires `Cube.initSolver()` once (slow table build) before `solve()`.

- **Scan is corner-on: two captures** (`CORNER_CAPTURES` in `detection.js`). The
  user points a cube *corner* at the camera so three faces show at once as
  foreshortened rhombi; two opposite corners cover all six faces. Geometry is a
  real **perspective projection** of a 3D cube (`computeCornerRegion` rotates a
  unit cube corner-on and projects through a pinhole camera), not a flat hexagon —
  the `persp` arg (0..1, a camera-distance slider in the UI) tapers the template
  and the sampling grid to match how close the cube is held; `persp=0` is the
  near-orthographic regular hexagon. Projecting the real sticker centres (per
  `FACE_AXES`) emits each face's 9 samples *already in facelet row-major order*, so
  everything downstream is unchanged. The second capture is reached by
  **one** unambiguous motion — a 180° flip about the horizontal screen axis
  (derived: the URF→DLB pose rotation is `diag(1,−1,−1)`); a 180° flip has no
  directional ambiguity, which is the whole reason for this design over the old six
  rotation instructions. `test/corner-geometry.test.mjs` guards the derivation
  against hardcoded Kociemba corner/edge adjacency (the three faces meet at one
  real cube corner; stickers along each shared rhombus edge glue correctly). What
  no offline test can prove — that the *physical holding instruction* matches
  reality (e.g. "F lower-left" really is F) — needs a real cube in the browser.
  The old flat-on scan (`SCAN_STEPS` in `cube-state.js`, `sampleGrid`,
  `scan-geometry.test.mjs`) is left in place but inactive, as a potential fallback.

- **Scan reorientation is shown by an HTML/SVG cube glyph** (`src/glyph.js`) in an
  un-mirrored overlay, NOT on the canvas — so it reads as a physical "hold/flip the
  cube this way" instruction independent of the preview mirror. (The solve arrows,
  by contrast, are on the canvas and intentionally flip with the cube.) The two
  corner glyphs are `corner` (bare corner-on cube) and `flip` (cube + 180° vertical
  flip arrow); keep them, the `CAPTURE_UI` hint text in `main.js`, and the geometry
  in `CORNER_CAPTURES` in sync. Hints name faces by the held corner or scanned
  colour, never left/right of a face, which would flip under the mirror.

- **Arrow directions** in `overlay.js` (`ARROWS`) are derived for the fixed frame
  "front center toward camera, top center up": `U`→top row LEFT, `D`→bottom row
  RIGHT, `R`→right col UP, `L`→left col DOWN, `F`→clockwise, `B`→counter-clockwise
  (as seen from the front). `'` inverts; `2` is the same direction twice. Centers
  never move under face turns, so these stay valid for the whole solve as long as
  the user keeps that orientation.

- **Color classification is balanced, center-anchored k-means** (`classifyFaces`
  in `colors.js`), not fixed thresholds. It clusters all 54 samples in CIE-Lab with
  k=6, the six captured centers pinned to their own clusters and every cluster
  forced to exactly nine — both strong cube priors. This beats plain nearest-center
  at the red↔orange boundary under warm light: clearly-colored stickers claim their
  nine slots and the ambiguous ones fall to the only label left. It returns
  `conflicts` (stickers the balance constraint pulled off their nearest centroid) =
  a residual-ambiguity signal. Because counts are 9-each *by construction*, that
  signal — not a count mismatch — is what flags a shaky read (see next bullet).
  `buildReferences`/`classify` (single nearest-center) remain for reference/tests.

- **Shaky scans fold in more passes instead of failing.** If a completed pass (both
  corners) fails to validate, has `conflicts > 0`, or doesn't solve, `main.js`
  re-scans up to `MAX_PASSES` times and `aggregateFaces` (`cube-state.js`) averages
  each facelet's RGB across all passes before re-classifying — the user holds the
  cube at a slightly different angle each pass so per-sticker glare/lighting
  averages out. Facelet indices are identical across passes (deterministic capture
  geometry), so averaging needs no registration. Only a final failure surfaces an
  error; an unresolved ambiguity after the last pass commits the best guess.

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

This is a **guided** corner-on scan: the user aligns a cube corner to a drawn
hexagon guide and we sample fixed warped positions — no cube detection yet. Known
gaps / next steps:

- The two captures are *antipodal* (opposite corners) so they share no faces;
  registration relies on the user's 180° flip being clean. A 3-capture variant of
  *overlapping* corners would cover all six faces with shared faces between shots,
  letting the overlap auto-resolve the orientation (and is the natural step toward
  real pose estimation).
- `detection.js` is still the seam for true cube tracking (contour detection +
  6-DoF pose via `solvePnP`, 3D arrows with three.js), which would drop the
  align-to-guide requirement entirely.
- Classification is balanced k-means over all 54 samples (done). A possible next
  step is weighting each sticker by sampling confidence (patch variance / glare),
  since oblique corner-on faces vary in quality more than flat-on did.

The pixel-sampling layer is the only part `npm test` can't cover — **verify camera
changes in a real browser with a physical cube.**
