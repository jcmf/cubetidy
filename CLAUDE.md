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
- A **line-based detector** (Canny + probabilistic Hough) is being explored as an
  alternative to the quad/contour path. `detectLineSegments` (`detect.js`) returns
  raw segments; `cv-worker.js` dispatches on `opts.method`, returning
  `{type:'segments'}` for hough vs `{type:'quads'}`. `tools/hough-image.mjs` renders
  the same pipeline on a still frame (`bg=black` to hide the cube, `raw=1` for
  ungrouped segments).
  - **Synthetic ground truth** (`src/synth.js` core; `tools/synth-cube.mjs`,
    `npm run synth`): builds a cube at a CHOSEN 6-DoF pose and draws it, reusing the
    detector's own `project()`/`GRID_OFFSETS` so image and truth can't drift. The CLI
    writes a PNG + a `.truth.json` sidecar (the known pose, `K`, per-sticker colours,
    projected corners/cells). This is how the PIXEL layer gets an offline test it
    otherwise can't have: real `samples/` frames have no ground truth.
    `tools/synth-smoke.mjs` (`npm run synth:check`) renders several poses, runs the
    REAL `detectLineSegments`→`solveCubeFromLines`, and grades the recovered pose
    against truth (lock + rotation mod the 24 symmetries + centre + depth). Loads the
    OpenCV WASM, so like `detect-smoke` it is NOT part of `npm test`. CLI outputs go to
    `samples/synth/` (git-ignored); `hough-image.mjs` points straight at them.
    `tools/synth-bench.mjs` (`npm run synth:bench`, ~18s) scales that into a MEASURING
    benchmark (smoke = pass/fail gate, bench = rates/medians): a deterministic
    100-scene matrix (5 views × 4 dists × clean/blur/noise/soft/scramble looks, seeded
    pose jitter), graded into tiers derived from true geometry — `core` (3 faces,
    dist 3–6, expect accurate lock), `edge` (2 faces), `far` (dist 9, out of regime:
    only FALSE locks count against). Adds a cell-hit-rate end-task metric ("would the
    colour reader sample the right sticker"). Writes JSON; `baseline=<prev.json>`
    diffs per-scene outcomes and exits 1 on regressions; problem scenes print a
    `synth-cube.mjs` repro command (`dump=1` also writes their PNGs); unknown
    `key=val`s pass through to the detector, so a knob's aggregate effect is one run.
    First find (2026-06): `imgBlur=2` triggered one-cell translation aliasing on ~10
    scenes — fixed by the step-3e adaptive Canny retry; scenes that locked via the
    retry are marked `*`. Remaining in-regime gap: rare wrong-R locks under heavy
    blur+noise (`tilted-d4.5-soft`). Bench and smoke also grade the COLOUR READER
    (step 4) under the recovered truth-aligned pose: `colorRate` = precision over the
    cells the reader CLAIMS (weight > 0), `colorCover` = claimed fraction — split that
    way because a weight-0 cell is a declared no-read (grazing face), not a misread;
    only scrambled scenes catch one-cell slips (solved looks read the face's own
    colour either way). The
    scene/draw core is BROWSER-SAFE (no skia/fs — those stay in the CLI) and shared
    with the in-page **`?synth`** harness (`main.js`): a synthetic cube replaces the
    camera as the frame source, with a tuning panel on the LEFT (pose/appearance
    sliders, URL-synced like `?detect`) and a Save-PNG+JSON button. `?synth` composes
    with the detector — a "Run line detector" toggle (or `?synth&detect`) overlays the
    real hough pipeline on the generated frame AND shows the detect tuning panel on the
    RIGHT, so you tune the detector while watching it lock onto a KNOWN pose. The two
    panels' URL-syncs are disjoint (each manages only its own schema keys + preserves
    the other's), so the synth knob `blur` is named `imgBlur` to avoid colliding with
    the detector's Canny `blur`; in `?synth` the overlay is tracked by `runDetect`.
  - **Step 1 (done): vanishing-point grouping** (`src/lines.js`,
    `test/lines.test.mjs`). A cube's edges run along 3 orthogonal directions, each
    projecting to a vanishing point; `groupLineSegments` splits the segments into 3
    families + their VPs (orientation-seeded k-means → EM that reassigns each line to
    the VP it points at and refits each VP as the smallest eigenvector of Σℓℓᵀ, with
    random restarts and Hartley normalization; clutter that points at no VP is an
    outlier). `drawLineGroups` (`overlay.js`) colours by family + draws VP
    crosshairs; `?detect&method=hough` shows it live, tunable via `vpMaxErrorDeg` /
    `vpIters`. Works well when the cube dominates the frame; a *convergent background
    bundle* (e.g. books) can still steal one of the 3 slots (fixed k=3) — clutter
    rejection / cube-region focus is for step 2.
  - **Step 2 (done): VPs → rotation** (`estimateRotationFromLines` in `lines.js`).
    RANSAC for three MUTUALLY-ORTHOGONAL vanishing directions = the rotation R: two
    lines give d1, a third (⊥ d1) gives d2, d3=d1×d2; the triple explaining the most
    inlier line length wins, refined on its inliers. Orthogonality is the cube prior
    that rejects clutter (a convergent background bundle isn't orthogonal to the
    cube's other axes, so it can't join the frame) — fixes the corner1 books case.
    Inlier error is the 3D angle between a line's back-projected plane and a direction
    (asin|n̂·d̂|, n̂=Kᵀℓ), NOT the 2D midpoint→VP angle, which over-collects toward a
    far VP. Two degeneracy guards matter: reject any axis within ~25° of the optical
    axis (a VP at the principal point is a false attractor that vacuums up central
    lines), and require all three axes to have ≥2 inliers (so a single attractor
    can't win on total weight); the refit is re-checked against both and reverts if it
    drifts. R is only up to the cube's 24 symmetries (fine for an orientation overlay;
    disambiguated later for the solve). `drawCubeWireframe` (`overlay.js`) shows it at
    a rough pose (R + translation guessed from inlier centroid/spread) — visible at
    `?detect&method=hough`, knob `vpRansac`. KEY LIMIT (real frames): VPs need
    perspective — works when the cube is reasonably large (corner2, solved-hand);
    unreliable on small/distant cubes (corner1, b-series) where projection is near-
    affine and direction depth is poorly constrained. A confidence gate is the natural
    follow-up.
  - **Step 3 (done): metric 6-DoF pose + confidence gate** (`recoverCubePose` in
    `lines.js`). With R fixed, project the cube grid model, associate each lattice
    corner to the nearest detected line of each in-plane family (point-to-SEGMENT, so a
    stray clutter line elsewhere can't shadow it), intersect to a measured corner, and
    feed those 3D↔2D pairs to the existing `refinePnP` (ICP, robust per-pass trim). The
    cube centre is symmetry-invariant so t is unambiguous despite the 24-fold R. Two
    things were essential: (1) seed depth with a 1-D SCAN that slides the cube along the
    centroid ray and matches projected grid lines to detected ones — the 3×3 grid is
    periodic, so a length-based depth (biased by foreshortening) lets ICP lock onto a
    wrong scale alias; (2) the gate: `locked` only when ≥`minCorr` corners reproject
    within `maxReprojFrac`·edge (knobs in the panel). A wrong/weak R yields few corners
    or a loose fit and does NOT lock, so the overlay shows a bright wireframe only when
    trustworthy (dim grey rough wireframe otherwise). Verified on synthetic and on real
    close-held frames (`samples/close1-3` lock with a correct wireframe; `corner2` /
    `solved-hand` / `scrambled-hand` lock; far/weak `corner1` / `corner3` correctly
    don't). Two fixes were needed for real frames: (a) seed depth from the inlier
    ENDPOINT SPREAD, not segment length — Hough fragments a close cube's grid so length
    under-reads its size and places it far too deep (outside the depth-scan band);
    (b) `solveCubeFromLines` SWEEPS the inlier angle (`vpSweep` = 3–6°) and keeps the
    best lock — the angle yielding a good orthogonal frame varies frame-to-frame and no
    single value is reliable (3° drops `close2`, 5° drops `close1`/`corner2`). A wrong
    frame fails to lock at every angle, so the sweep can't manufacture a false lock.
    Live and offline both call `solveCubeFromLines`; an explicit `vpMaxErrorDeg` from
    the panel pins the sweep to one value.
  - **Step 3b (done): temporal smoothing** (`smoothLinePose` in `lines.js`). The
    per-frame lock is correct most frames but occasionally snaps wrong, and even correct
    consecutive frames jitter (~±8° rotation, ~±25% depth on the close samples). The
    smoother EMA-blends across frames but must respect the 24-fold symmetry: it
    `canonicalizeRotation`s each new R to the symmetry representative nearest the
    smoothed one (else blending two valid-but-different reps corrupts the pose), blends
    if consistent (within `poseGateAngle` / `poseGateTrans`), and REJECTS jumps — the
    intermittent wrong locks — unless a new pose persists `poseReacquire` updates (a real
    move). Holds through `poseMaxMiss` lock-less updates. `CUBE_ROTATIONS`,
    `canonicalizeRotation`, `rotationAngleDeg`, `blendRotation` are the pure pieces;
    `smoothLinePose` is pure too (state in/out) and tested with a synthetic sequence
    (holds through wrong locks, re-acquires on a sustained move, releases on dropout).
    Panel knobs: `poseAlpha` / `poseGateAngle` / `poseReacquire`.
  - **Step 3c (done): kill the translation aliasing — recover t WITHOUT corner PnP.**
    The lock was right-angle but wrong-size/position and flickered by whole stickers,
    because the 3×3 grid is PERIODIC: a ⅔-scaled / cell-shifted cube reprojects nearly as
    well, and ANY corner-reprojection objective (`refinePnP`, grid-match) actively PREFERS
    the alias (it fits a tight subset). Tellingly, reprojection was anti-correlated with
    correctness across frames (the lowest-error frame was the most wrong). So translation
    is NOT recovered by PnP at all — it's recovered from two alias-free image statistics:
      1. the line field's overall EXTENT → depth/scale (anchorT scales `t` along its ray;
         scaling `t` along its own ray leaves the projected CENTRE fixed, only resizing);
      2. its CENTROID → lateral position (anchorT shifts `t` laterally).
    Both use the visible-face grid-line endpoints (model and detected alike, so the
    corner-on "front faces only" bias cancels) with a ROBUST bbox/centroid (median centre,
    drop endpoints beyond 3× median radius — a stray clutter line in a family otherwise
    blows a plain bbox up ~4×). That anchor picks the right CELL but is coarse; it's then
    sharpened by a local search maximizing soft COVERAGE — detected line length lying ON a
    projected grid line (ramped by distance). Coverage is the one objective NOT fooled by
    the aliases: a shifted/shrunk cube leaves the OUTER detected lines uncovered, lowering
    it. `solveCubeFromLines` likewise selects the sweep's best lock by COVERAGE, never
    reprojection. R still comes from the VPs (unchanged). Result on the `ceiling1-6`
    samples: all lock tight (reproj ~2-6% of edge) and consistent (edge spread 18%→6%),
    grid overlays the detected lines. Knob `minCover`. Verify live steadiness/accuracy.
  - **Step 3e (done): adaptive Canny retry for soft frames** (`detectAndSolveLines` in
    `detect.js` — now THE entry point, shared by the worker and all offline tools; the
    WORKER also runs the solve now and posts `sol` with the segments, so the retry can
    re-run detection and the RANSAC cost leaves the UI thread — `main.js` reads
    `latestLineSol()`). Found by synth-bench: on blurred/defocused frames the crisp
    Canny pair (40/120) loses the thin inter-sticker gap edges on oblique faces, ONE
    VP family starves (min/max family length ≤0.32 on every aliased lock), and the
    periodic grid CONFIDENTLY locks a cell sideways — while 20/60 everywhere floods
    cluttered crisp frames and breaks their locks (corner2). Neither scoring tweak nor
    margin tuning fixes this geometrically (measured: per-family-normalized coverage
    does NOT separate the alias — the 8-line model comb at tol 0.22·cell is too dense),
    so the fix is at DETECTION level: re-detect at `retryCannyLo/Hi` (20/60) when the
    crisp solve is suspect (no lock, or family balance < `retryBalance`), then hold the
    retry to a harder standard than the gate: a retry-ONLY lock needs cover ≥
    `retryMinCover` (0.7 — wrong soft locks measured ≤0.68, real ones ≥0.74); when BOTH
    passes lock, `gridCoverScore` (exported, lines.js, tested) cross-evaluates the two
    POSES on the union of both segment sets (same evidence ⇒ clutter dilution cancels;
    the aliased pose misses the recovered gap lines) and the retry must win by
    `retryMargin`. Result: bench in-regime false locks 10→1 (the 1 is a wrong-R case,
    a different failure class), all real-frame locks/non-locks unchanged (ceiling3's
    retry even pulls its edge estimate in line with its burst siblings).
  - **Step 4 (done): read sticker colours off the locked grid** (`src/read-colors.js`,
    pure + browser-safe). `readStickerColors(imageData, K, pose)` projects each
    genuinely-visible face's nine cell centres (perspective-correct `visibleCubeFaces`,
    shared from pose.js with the synth renderer), samples a 5×5 patch of real 3D
    points inside the inner half of each sticker, and reduces to a median RGB plus a
    confidence `weight` (view cosine × in-frame fraction ÷ patch scatter) — a grazing
    face or glare boundary weighs ~0 instead of misreading, so PRECISION over claimed
    cells is the quality bar and coverage is a separate number. Faces are keyed by
    BODY-frame axis (k ∈ 0..2, s = ±1), cells row-major exactly like buildCubeScene's
    truth facelets, so a truth-aligned (`canonicalizeRotation`'d) pose grades
    index-for-index; which face LETTER a (k,s) is stays unknown until the six centre
    colours are classified (the 24-fold R ambiguity). `accumulateStickerColors` (pure,
    state in/out like smoothLinePose) weighted-means each sticker across frames —
    view-dependent glare averages out; keys are only stable while the lock is
    continuous, so main.js clears it when the smoother releases or re-acquires.
    TESTING: `test/read-colors.test.mjs` runs in `npm test` against REAL rendered
    pixels — drawScene needs only createImageData/putImageData, so a stub context
    rasterizes scenes in plain node and the reader samples at the TRUTH pose (the
    scrambled scene pins the indexing against transposition/mirroring); reads under a
    RECOVERED pose are graded by smoke + bench (in-regime medians 100%@100%). LIVE:
    `?detect&method=hough` / `?synth&detect` draw raw-RGB swatches on the locked cells
    (`drawCellColors`); the read happens in requestDetectFrame BEFORE the frame buffer
    is transferred to the worker (transfer detaches it); under ?synth the status line
    grades claimed letters against truth live.
  - **Next:** assemble the facelet string: classify the accumulated 27-per-corner
    colours (balanced k-means, colors.js) once coverage suffices, identify faces by
    their CENTRE colours (resolves the 24-fold R and registers the two opposite-corner
    captures without trusting the user's flip), validate + solve.
- The `?detect` harness has an **on-page tuning panel** (`buildDetectPanel` in
  `main.js`, `#detect-panel` styles) so the detector's knobs are sliders + a method
  dropdown, not query-string edits. Sliders are schema-driven (`DETECT_PARAMS`,
  method-filtered) and mutate `detectOpts` in place (the worker reads it every
  frame); changes also `history.replaceState` back into the URL, so a copied link or
  reload restores the tuning. A "Hide camera (lines only)" toggle blacks out the
  frame *after* it's sampled for detection (so detection still sees the real cube).
  To add a knob: add a `DETECT_PARAMS` row — no other wiring needed.
- Classification is balanced k-means over all 54 samples (done). A possible next
  step is weighting each sticker by sampling confidence (patch variance / glare),
  since oblique corner-on faces vary in quality more than flat-on did.

The pixel-sampling layer is the only part `npm test` can't cover — **verify camera
changes in a real browser with a physical cube.**
