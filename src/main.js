import { startCamera } from './camera.js';
import { computeRegion, computeCornerRegion, sampleCorner, CORNER_CAPTURES } from './detection.js';
import { toFaceletString, validate, aggregateFaces } from './cube-state.js';
import { initSolver, solve } from './solver.js';
import { drawGuide, drawCornerGuide, drawMove, drawDetections, drawCube, drawSegments, drawLineGroups, drawCubeWireframe } from './overlay.js';
import { glyphSVG } from './glyph.js';
import { startCV, cvReady, requestDetect, latestQuads, latestSegments } from './opencv.js';
import { findFaceGrids, fitFaceGrid } from './group.js';
import { estimateCubePose } from './cube-pose.js';
import { estimateIntrinsics, project } from './pose.js';
import { DETECT_DEFAULTS, HOUGH_DEFAULTS } from './detect.js';
import { groupLineSegments, solveCubeFromLines, smoothLinePose, canonicalizeRotation, rotationAngleDeg, VP_DEFAULTS, ROT_DEFAULTS, POSE_DEFAULTS, POSE_SMOOTH_DEFAULTS } from './lines.js';
import { buildCubeScene, drawScene } from './synth.js';

// Diagnostic harness for the OpenCV detector: open with ?detect to overlay
// detected sticker quads on the live frame while tuning against a real cube.
// Detection runs in a worker; this is pure visualization, not wired into the
// scan state machine yet.
const DEBUG_DETECT = new URLSearchParams(location.search).has('detect');
// ?synth: render a synthetic cube (KNOWN pose/colours) into the canvas in place of
// the camera — an in-page test-image generator you can eyeball, tune, and save.
// Composable with the detector: a "Run line detector" toggle (or ?synth&detect)
// overlays the REAL hough pipeline on the generated frame, so you watch it lock onto
// a known pose. The scene/draw core is shared with the node CLI via src/synth.js.
const DEBUG_SYNTH = new URLSearchParams(location.search).has('synth');
const detectState = { frame: 0, lastStatus: '', lastQuads: null, quadCount: 0, cube: null, lastSegments: null, grouping: null, sol: null, smoothPose: null, K: null };
// Every query param besides the mode flags overrides a DETECT_DEFAULTS knob, so
// detection can be tuned live from the URL without a rebuild, e.g.
//   ?detect&method=adaptive&blockSize=51&C=7   or   ?detect&cannyLo=20&minFill=0.4
const MODE_FLAGS = new Set(['detect', 'synth', 'runDetect']);
const detectOpts = (() => {
  const o = {};
  for (const [k, v] of new URLSearchParams(location.search)) {
    if (MODE_FLAGS.has(k)) continue;
    o[k] = /^-?\d*\.?\d+$/.test(v) ? parseFloat(v) : v;
  }
  return o;
})();
// Display-only state for the panel (not a worker knob). `hideCamera` blacks out the
// frame so only the detection overlay shows; pulled out of detectOpts so it isn't
// shipped to the worker, but it still round-trips through the URL like the rest.
const detectDisplay = { hideCamera: false };
if ('hideCamera' in detectOpts) { detectDisplay.hideCamera = !!detectOpts.hideCamera; delete detectOpts.hideCamera; }

// ?synth state. synthOpts holds the scene + appearance knobs (any URL param besides
// the mode flags), merged over SYNTH_DEFAULTS each frame so untouched DOFs keep their
// defaults (esp. the rotation axis — a single touched axis slider mustn't zero the
// other two). runDetect overlays the line detector (implied by ?synth&detect).
const synthOpts = (() => {
  const o = {};
  for (const [k, v] of new URLSearchParams(location.search)) {
    if (MODE_FLAGS.has(k)) continue;
    o[k] = /^-?\d*\.?\d+$/.test(v) ? parseFloat(v) : v;
  }
  return o;
})();
const synthDisplay = { runDetect: DEBUG_SYNTH && (new URLSearchParams(location.search).has('runDetect') || DEBUG_DETECT) };
const synthState = { scene: null };
const SYNTH_DEFAULTS = { angleDeg: 57, axisX: 0.9, axisY: -1, axisZ: 0.1, dist: 6, tx: 0, ty: 0, gap: 0.1, imgBlur: 0, noise: 0, scramble: 0 };
// Slider schema for the ?synth panel (bounds only; values come from SYNTH_DEFAULTS).
const SYNTH_PARAMS = [
  { k: 'angleDeg', label: 'Angle°',    min: 0,  max: 180, step: 1 },
  { k: 'axisX',    label: 'Axis X',    min: -1, max: 1,   step: 0.05 },
  { k: 'axisY',    label: 'Axis Y',    min: -1, max: 1,   step: 0.05 },
  { k: 'axisZ',    label: 'Axis Z',    min: -1, max: 1,   step: 0.05 },
  { k: 'dist',     label: 'Distance',  min: 3,  max: 12,  step: 0.1 },
  { k: 'tx',       label: 'Offset X',  min: -2, max: 2,   step: 0.05 },
  { k: 'ty',       label: 'Offset Y',  min: -2, max: 2,   step: 0.05 },
  { k: 'gap',      label: 'Sticker gap', min: 0, max: 0.3, step: 0.01 },
  { k: 'imgBlur',  label: 'Blur px',   min: 0,  max: 8,   step: 0.5 },
  { k: 'noise',    label: 'Noise',     min: 0,  max: 40,  step: 1 },
  { k: 'scramble', label: 'Scramble seed', min: 0, max: 60, step: 1 },
];
if (DEBUG_SYNTH && synthDisplay.runDetect) detectOpts.method ??= 'hough'; // hough by default, but honour an explicit ?method=

// Schema for the on-page tuning panel: every numeric detector knob with its slider
// bounds and which method(s) it applies to (only the active method's knobs show).
// Defaults come from DETECT_DEFAULTS / HOUGH_DEFAULTS; this just bounds the sliders.
const currentMethod = () => detectOpts.method ?? 'canny';
const DETECT_PARAMS = [
  // shared by the canny line/quad preprocessing
  { k: 'blur',        label: 'Blur (odd)',     min: 1,      max: 21,  step: 2,      methods: ['canny', 'hough'] },
  { k: 'cannyLo',     label: 'Canny low',      min: 0,      max: 255, step: 1,      methods: ['canny', 'hough'] },
  { k: 'cannyHi',     label: 'Canny high',     min: 0,      max: 300, step: 1,      methods: ['canny', 'hough'] },
  // hough line explorer
  { k: 'houghThresh', label: 'Hough votes',    min: 1,      max: 200, step: 1,      methods: ['hough'] },
  { k: 'minLineLen',  label: 'Min line len',   min: 0,      max: 300, step: 1,      methods: ['hough'] },
  { k: 'minLineFrac', label: 'Min line frac',  min: 0,      max: 0.3, step: 0.005,  methods: ['hough'] },
  { k: 'maxLineGap',  label: 'Max line gap',   min: 0,      max: 50,  step: 1,      methods: ['hough'] },
  // vanishing-point grouping (step 1) + orthogonal-frame rotation search (step 2)
  { k: 'vpMaxErrorDeg', label: 'VP max err°',  min: 0.5,    max: 15,  step: 0.5,    methods: ['hough'] },
  { k: 'vpIters',     label: 'VP iters',       min: 1,      max: 12,  step: 1,      methods: ['hough'] },
  { k: 'vpRansac',    label: 'VP RANSAC',      min: 50,     max: 2000, step: 50,     methods: ['hough'] },
  // metric pose + confidence gate (step 3)
  { k: 'minCorr',     label: 'Min corners',    min: 4,      max: 30,  step: 1,      methods: ['hough'] },
  { k: 'maxReprojFrac', label: 'Max reproj',   min: 0.02,   max: 0.3, step: 0.01,   methods: ['hough'] },
  // temporal smoothing of the locked pose (step 3b)
  { k: 'poseAlpha',   label: 'Smooth α',       min: 0.1,    max: 1,   step: 0.05,   methods: ['hough'] },
  { k: 'poseGateAngle', label: 'Jump gate°',   min: 5,      max: 45,  step: 1,      methods: ['hough'] },
  { k: 'poseReacquire', label: 'Reacquire',    min: 1,      max: 15,  step: 1,      methods: ['hough'] },
  // canny quad detector
  { k: 'dilateIters', label: 'Dilate iters',   min: 0,      max: 5,   step: 1,      methods: ['canny'] },
  { k: 'closeIters',  label: 'Close iters',    min: 0,      max: 5,   step: 1,      methods: ['canny'] },
  // mask quad detector
  { k: 'satThresh',   label: 'Sat threshold',  min: 0,      max: 255, step: 1,      methods: ['mask'] },
  { k: 'valThresh',   label: 'Val threshold',  min: 0,      max: 255, step: 1,      methods: ['mask'] },
  // shared quad gates
  { k: 'approxEps',   label: 'Approx eps',     min: 0.01,   max: 0.2, step: 0.005,  methods: ['canny', 'mask'] },
  { k: 'minAreaFrac', label: 'Min area frac',  min: 0.0001, max: 0.01, step: 0.0001, methods: ['canny', 'mask'] },
  { k: 'maxAreaFrac', label: 'Max area frac',  min: 0.005,  max: 0.3, step: 0.005,  methods: ['canny', 'mask'] },
  { k: 'minFill',     label: 'Min fill',       min: 0,      max: 1,   step: 0.01,   methods: ['canny', 'mask'] },
  { k: 'maxAspect',   label: 'Max aspect',     min: 1,      max: 6,   step: 0.1,    methods: ['canny', 'mask'] },
  { k: 'medianLo',    label: 'Median lo',      min: 0,      max: 1,   step: 0.01,   methods: ['canny', 'mask'] },
  { k: 'medianHi',    label: 'Median hi',      min: 1,      max: 10,  step: 0.1,    methods: ['canny', 'mask'] },
];
// Temporal smoothing of the cube pose (overridable via the URL detectOpts).
const SMOOTH = {
  alpha: detectOpts.smoothAlpha ?? 0.35, // EMA weight toward each new estimate (lower = smoother)
  maxMiss: detectOpts.maxMiss ?? 6,      // hold the last good pose this many updates through dropouts
  minScore: detectOpts.minScore ?? 6,    // ignore weak estimates (few matched stickers = likely wrong)
  gateFrac: detectOpts.gateFrac ?? 0.4,  // a new estimate must land within this * cube radius of the
                                         //   locked one (held still), else it's a jump (false lock)
  reacquire: detectOpts.reacquire ?? 8,  // ...persisting this many updates = a real move; re-lock
};
if (DEBUG_DETECT) console.log('[detect] debug overlay ON; opts =', detectOpts,
  '— start the camera. Click the preview (or press "c") to download the current frame as test data.');

// Per-capture UI copy for the corner-on scan. Geometry lives in CORNER_CAPTURES;
// hints name the held corner relative to the previous one (never left/right of a
// face, which would flip under the preview mirror).
const CAPTURE_UI = [
  { glyph: 'corner',
    hint: 'Point a <b>corner</b> of the cube straight at the camera and line it up ' +
      'with the outline — its three faces fill the three diamonds.' },
  { glyph: 'flip',
    hint: 'Flip the cube 180° about the <b>left–right</b> axis to bring the opposite ' +
      'corner forward — follow the arrow. Keep the same side on your left.' },
];

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

const els = {
  status: document.getElementById('status'),
  hint: document.getElementById('hint'),
  captured: document.getElementById('captured'),
  primary: document.getElementById('primary'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  reset: document.getElementById('reset'),
  mirror: document.getElementById('mirror'),
  glyph: document.getElementById('glyph'),
  perspective: document.getElementById('perspective'),
  perspectiveControl: document.getElementById('perspective-control'),
};

// Up to this many full passes (each pass = both corners) before giving up. Extra
// passes only happen when a pass fails to validate/solve; their per-sticker
// readings are averaged in to ride out lighting/glare (esp. red vs orange).
const MAX_PASSES = 3;

const state = {
  phase: 'idle', // idle | scanning | solving | guide | error
  scanIndex: 0,
  persp: parseFloat(els.perspective.value), // corner-guide perspective (0..1)
  pass: 1,        // current scan pass (1-based)
  passes: [],     // completed passes; each is { letter: [9 samples] }
  faces: {},      // the in-progress pass's accumulated faces
  solution: [],
  moveIndex: 0,
};

// --- render loop -----------------------------------------------------------

function render() {
  if (DEBUG_SYNTH) { renderSynth(); requestAnimationFrame(render); return; }
  if (video.videoWidth && video.videoHeight) {
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Sample the CLEAN frame for detection here — before any guide/arrow overlay is
    // drawn onto the canvas, or the detector finds our own lines, not the cube.
    if (DEBUG_DETECT) requestDetectFrame();

    // "Lines only" view: black out the frame AFTER it's been sampled for detection,
    // so the detector still sees the real cube but the overlay draws on black.
    if (DEBUG_DETECT && detectDisplay.hideCamera) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    if (state.phase === 'scanning' && !DEBUG_DETECT) {
      // The hold-the-cube guide just gets in the way of auto-detection.
      const scene = computeCornerRegion(canvas.width, canvas.height, state.persp, state.scanIndex);
      drawCornerGuide(ctx, scene, true);
    } else if (state.phase === 'guide') {
      const region = computeRegion(canvas.width, canvas.height);
      drawGuide(ctx, region, false);
      if (state.moveIndex < state.solution.length) {
        drawMove(ctx, region, state.solution[state.moveIndex]);
      } else {
        drawSolved(region);
      }
    }

    if (DEBUG_DETECT) drawDetectResults();
  }
  requestAnimationFrame(render);
}

// --- ?synth: in-page synthetic test-image generator -------------------------

// Effective scene/appearance opts: defaults under the live overrides.
const synthEffective = () => ({ ...SYNTH_DEFAULTS, ...synthOpts });

// Build + draw the synthetic cube into the canvas each frame. When runDetect is on,
// the freshly-drawn (overlay-free) frame is sampled for the detector and the hough
// wireframe is overlaid on top — letting you watch the real detector lock onto a
// KNOWN pose. K matches buildCubeScene's (same default fov), so it sees true ground truth.
function renderSynth() {
  const o = synthEffective();
  const W = +o.width || 1280, H = +o.height || 720;
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const scene = buildCubeScene({ width: W, height: H, ...o });
  drawScene(ctx, scene, o);
  synthState.scene = scene;
  if (synthDisplay.runDetect) {
    requestDetectFrame();  // samples the clean synthetic frame (no overlays drawn yet)
    drawDetectResults();   // overlays the hough grouping + wireframe + status
  } else {
    setDetectStatus(`synth: faces <b>${scene.truth.visibleFaces.map((f) => f.letter).join('')}</b> · `
      + `${scene.truth.angleDeg.toFixed(0)}° · dist ${scene.truth.dist} · edge ${scene.truth.edgePx.toFixed(0)}px`);
  }
}

// Download a data URL as a file, inside the current user gesture (sync, so the
// browser doesn't block the download — same trick as captureFrame).
function downloadDataURL(href, name) {
  const a = document.createElement('a');
  a.href = href; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

// Save the CLEAN generated frame (re-rendered without any detector overlay) + its
// ground-truth JSON, the browser-side equivalent of the node CLI's PNG + .truth.json.
function saveSynth() {
  const o = synthEffective();
  const W = canvas.width, H = canvas.height;
  const scene = buildCubeScene({ width: W, height: H, ...o });
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  drawScene(off.getContext('2d'), scene, o);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  downloadDataURL(off.toDataURL('image/png'), `synth-${stamp}.png`);
  downloadDataURL('data:application/json,' + encodeURIComponent(JSON.stringify(scene.truth, null, 2)), `synth-${stamp}.truth.json`);
  console.log(`[synth] saved synth-${stamp}.png (${W}x${H}) + truth JSON`);
}

// Ship the clean frame to the detection worker every few frames (it reads the
// whole frame). Called BEFORE overlays are drawn, so the sampled pixels are the
// raw camera image with none of our guide/arrow lines on them.
function requestDetectFrame() {
  if (!cvReady()) return;
  if (detectState.frame++ % 3 === 0) {
    requestDetect(ctx.getImageData(0, 0, canvas.width, canvas.height), detectOpts);
  }
}

// ?synth only: grade a recovered pose against the scene's KNOWN truth (centre offset as
// a fraction of edge, rotation mod the 24 cube symmetries, depth). Returns '' outside
// ?synth (no truth) — same metrics as tools/synth-smoke.mjs so live and offline agree.
function truthError(pose) {
  const scene = synthState.scene;
  if (!scene || !pose) return '';
  const K = detectState.K, tp = scene.pose, ePx = scene.truth.edgePx || 1;
  const cT = project(K, tp, [0, 0, 0]), cF = project(K, pose, [0, 0, 0]);
  const centre = Math.hypot(cF[0] - cT[0], cF[1] - cT[1]) / ePx * 100;
  const rot = rotationAngleDeg(tp.R, canonicalizeRotation(pose.R, tp.R));
  const depth = Math.abs(pose.t[2] - tp.t[2]) / (tp.t[2] || 1) * 100;
  return ` · <b>vs truth</b> centre ${centre.toFixed(0)}% rot ${rot.toFixed(0)}° depth ${depth.toFixed(0)}%`;
}

// On each fresh detection result, group -> fit -> temporally smooth; draw the
// smoothed faces every frame. Count goes to the status bar, never the canvas
// (mirrored text reads backwards).
function drawDetectResults() {
  if (!cvReady()) { setDetectStatus('detect: loading OpenCV…'); return; }

  // method=hough: Canny + probabilistic Hough, then the line-based detector.
  //  · step 2 searches for three ORTHOGONAL vanishing points (= rotation R), which
  //    groups the segments and rejects clutter (a non-orthogonal background bundle
  //    can't join the frame);
  //  · step 3 recovers a metric pose (grid lattice -> PnP) and a confidence gate:
  //    LOCKED draws a bright refined wireframe; an unlocked R draws a dim, untrusted
  //    rough wireframe (so a wrong/weak-perspective estimate never looks confident).
  // Recompute only on new segments.
  if (detectOpts.method === 'hough') {
    const segments = latestSegments();
    if (segments !== detectState.lastSegments) {
      detectState.lastSegments = segments;
      detectState.K = estimateIntrinsics(canvas.width, canvas.height);
      detectState.sol = solveCubeFromLines(segments, detectState.K, detectOpts);
      detectState.grouping = detectState.sol ? detectState.sol.rot : groupLineSegments(segments, detectOpts);
      // Temporally fuse the locked pose (reject the intermittent wrong locks, hold
      // through dropouts) so the overlay is steady instead of flickering right/wrong.
      // sol.fit is the recoverCubePose result; the metric {R,t} is sol.fit.pose.
      const fit = detectState.sol && detectState.sol.fit;
      detectState.smoothPose = smoothLinePose(detectState.smoothPose, fit && fit.locked ? fit.pose : null, detectOpts);
    }
    const g = detectState.grouping, sol = detectState.sol, sm = detectState.smoothPose;
    const fit = sol && sol.fit;
    if (g && g.families.length) drawLineGroups(ctx, g);
    else drawSegments(ctx, segments);
    if (sm) drawCubeWireframe(ctx, detectState.K, sm, '#39ff14');
    else if (sol && sol.rot.pose) drawCubeWireframe(ctx, detectState.K, sol.rot.pose, 'rgba(150,160,175,0.55)');
    const counts = g && g.families.length ? g.families.map((f) => f.segments.length).join('/') : '—';
    // In ?synth the ground-truth pose is known, so grade the DISPLAYED (smoothed) lock
    // against it — the whole point of ?synth is watching the detector hit a known pose,
    // and this is the only place that closes the offline/live gap (the browser rasterizes
    // the cube differently than skia, so segments — and the lock — differ from synth-smoke).
    setDetectStatus(
      `detect[hough]: <b>${segments.length}</b> segs · families <b>${counts}</b>` +
      (sm ? ` · <b>LOCK</b>${fit && fit.locked ? ` ${fit.count}pts ${fit.reprojErr.toFixed(1)}px` : ' (held)'}` + truthError(sm)
        : sol ? ' · searching' : ' · no R'));
    return;
  }

  const quads = latestQuads();
  if (quads !== detectState.lastQuads) { // recompute only when the worker returns new quads
    detectState.lastQuads = quads;
    detectState.quadCount = quads.length;
    const K = estimateIntrinsics(canvas.width, canvas.height);
    const fits = findFaceGrids(quads, detectOpts).map((f) => fitFaceGrid(f, detectOpts)).filter(Boolean);
    detectState.cube = smoothCube(estimateCubePose(fits, K, quads, detectOpts));
  }
  drawDetections(ctx, quads);
  if (detectState.cube) drawCube(ctx, detectState.cube.faces);

  const cube = detectState.cube;
  setDetectStatus(
    `detect[${detectOpts.method || 'canny'}]: <b>${detectState.quadCount}</b> quads · ` +
    (cube ? `cube <b>${cube.faces.length}</b> faces · score ${cube.score}` : 'no cube'));
}

// Centroid + RMS radius of a cube's projected sticker cells (a pose-agnostic
// summary for the motion gate).
function cubeStat(faces) {
  const pts = faces.flatMap((f) => f.cells);
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const rad = Math.sqrt(pts.reduce((s, p) => s + (p.x - cx) ** 2 + (p.y - cy) ** 2, 0) / pts.length);
  return { cx, cy, rad };
}
const cloneFaces = (faces) => faces.map((f) => ({ normal: f.normal, cells: f.cells.map((p) => ({ ...p })) }));
const acquire = (cube) => ({ faces: cloneFaces(cube.faces), score: cube.score, stat: cubeStat(cube.faces), miss: 0, reject: 0 });

// Temporally fuse cube-pose estimates. The cube is held roughly still, so:
//  - weak estimates (score < minScore) are ignored (hold the last good pose);
//  - a good estimate that JUMPS far from the locked pose is rejected as a false
//    lock / bad frame — unless it persists, which means a real move (re-acquire);
//  - estimates near the locked pose are EMA-blended. This motion gate is what makes
//    the overlay steady: it sticks to the consensus and ignores intermittent garbage.
function smoothCube(cube) {
  const prev = detectState.cube;
  if (!cube || cube.score < SMOOTH.minScore) {
    return prev && prev.miss < SMOOTH.maxMiss ? { ...prev, miss: prev.miss + 1 } : null;
  }
  if (!prev || !prev.stat) return acquire(cube);

  const stat = cubeStat(cube.faces);
  const moved = Math.hypot(stat.cx - prev.stat.cx, stat.cy - prev.stat.cy) > SMOOTH.gateFrac * prev.stat.rad
    || stat.rad / (prev.stat.rad || 1) < 0.7 || stat.rad / (prev.stat.rad || 1) > 1.4;
  if (moved) {
    const reject = (prev.reject ?? 0) + 1;
    return reject < SMOOTH.reacquire ? { ...prev, miss: 0, reject } : acquire(cube);
  }
  if (prev.faces.length !== cube.faces.length) return acquire(cube);
  const a = SMOOTH.alpha;
  const faces = cube.faces.map((f, fi) => ({
    normal: f.normal,
    cells: f.cells.map((p, i) => {
      const q = prev.faces[fi].cells[i];
      return { x: q.x + a * (p.x - q.x), y: q.y + a * (p.y - q.y) };
    }),
  }));
  return { faces, score: cube.score, stat: cubeStat(faces), miss: 0, reject: 0 };
}

// Update the status bar only when the text changes (avoids per-frame DOM churn).
function setDetectStatus(html) {
  if (html === detectState.lastStatus) return;
  detectState.lastStatus = html;
  setStatus(html);
}

function drawSolved(region) {
  // Green flash only — no canvas text (it would read backwards when mirrored).
  // The "solved" message lives in the status bar.
  const { x, y, side } = region;
  ctx.save();
  ctx.fillStyle = 'rgba(52,199,89,0.22)';
  ctx.fillRect(x, y, side, side);
  ctx.lineWidth = Math.max(4, side * 0.02);
  ctx.strokeStyle = '#34c759';
  ctx.strokeRect(x, y, side, side);
  ctx.restore();
}

// --- UI helpers ------------------------------------------------------------

function setStatus(text) { els.status.innerHTML = text; }
function setHint(html) { els.hint.innerHTML = html; }

function addThumb(samples) {
  const t = document.createElement('div');
  t.className = 'thumb';
  for (const s of samples) {
    const i = document.createElement('i');
    i.style.background = `rgb(${s.r},${s.g},${s.b})`;
    t.appendChild(i);
  }
  // Mark only the newest as active.
  els.captured.querySelectorAll('.thumb.active').forEach((e) => e.classList.remove('active'));
  t.classList.add('active');
  els.captured.appendChild(t);
}

function showButtons({ primary, prev, next, reset }) {
  els.primary.hidden = primary == null;
  if (primary != null) els.primary.textContent = primary;
  els.prev.hidden = !prev;
  els.next.hidden = !next;
  els.reset.hidden = !reset;
}

// The move label that used to be drawn on the canvas now lives in the status bar.
function showCurrentMove() {
  const i = state.moveIndex, n = state.solution.length;
  const tok = state.solution[i];
  const note = tok[0] === 'B' ? ' · back layer' : '';
  setStatus(`Move ${i + 1}/${n} — <b>${tok}</b>${note}`);
  const chips = state.solution.map((m, k) => {
    const cls = k === i ? 'current' : k < i ? 'done' : '';
    return `<span class="move-chip ${cls}">${m}</span>`;
  }).join('');
  // Name the front/up faces by their scanned colour: after a corner scan the
  // U/F/R labels came from the held corner, so colours are how the user knows
  // which way to hold the cube for the arrows.
  setHint(
    `Hold the ${swatch('F')} centre toward you, ${swatch('U')} centre up.` +
    `<div class="move-list">${chips}</div>`
  );
}

// A small inline colour chip for a face's scanned centre.
function swatch(letter) {
  const c = state.faces[letter]?.[4];
  return c ? `<i class="swatch" style="background:rgb(${c.r},${c.g},${c.b})"></i>` : '';
}

// --- phase transitions -----------------------------------------------------

function enterScanning() {
  state.phase = 'scanning';
  state.scanIndex = 0;
  state.pass = 1;
  state.passes = [];
  state.faces = {};
  els.captured.innerHTML = '';
  promptCurrentScan();
}

function promptCurrentScan(note) {
  const cap = CORNER_CAPTURES[state.scanIndex];
  const ui = CAPTURE_UI[state.scanIndex];
  const pass = state.pass > 1 ? ` · pass ${state.pass}` : '';
  setStatus(`Scan ${state.scanIndex + 1}/${CORNER_CAPTURES.length} — <b>${cap.title}</b>${pass}`);
  setHint((note ? `<b>${note}</b><br>` : '') + ui.hint);
  setGlyph(ui.glyph);
  els.perspectiveControl.hidden = false;
  showButtons({ primary: 'Capture', reset: true });
}

// Show the corner-on instruction glyph ('corner'/'flip'); null hides it.
function setGlyph(motion) {
  const svg = glyphSVG(motion);
  els.glyph.innerHTML = svg;
  els.glyph.hidden = !svg;
}

function captureCorner() {
  const scene = computeCornerRegion(canvas.width, canvas.height, state.persp, state.scanIndex);
  const faces = sampleCorner(ctx, scene);
  for (const f of scene.faces) {
    state.faces[f.letter] = faces[f.letter];
    addThumb(faces[f.letter]);
  }

  state.scanIndex++;
  if (state.scanIndex < CORNER_CAPTURES.length) {
    promptCurrentScan();
  } else {
    state.passes.push(state.faces);
    solveScanned();
  }
}

async function solveScanned() {
  state.phase = 'solving';
  setStatus('Solving…');
  setHint('Computing the shortest solution.');
  setGlyph(null);
  els.perspectiveControl.hidden = true;
  showButtons({ primary: null, reset: true });

  const faces = aggregateFaces(state.passes);
  state.faces = faces; // aggregate drives the solve-orientation swatches
  const { facelets, counts, conflicts } = toFaceletString(faces);

  const check = validate(facelets, counts);
  const havePasses = state.passes.length < MAX_PASSES;
  if (!check.ok) return havePasses ? anotherPass() : failScan(check.error);

  // The balanced classifier had to override some stickers' nearest color, so the
  // read is ambiguous (typically red vs orange). Gather another angle before
  // committing while we still can.
  if (conflicts > 0 && havePasses) return anotherPass();

  let solution;
  try {
    solution = await solve(facelets);
  } catch (err) {
    const msg = `Unsolvable state — a color was misread. (${err.message})`;
    return havePasses ? anotherPass() : failScan(msg);
  }

  state.solution = solution;
  state.moveIndex = 0;
  state.phase = 'guide';
  if (state.solution.length === 0) {
    setStatus('✓ Already solved!');
    setHint('This cube is already in the solved state.');
    showButtons({ primary: 'Scan another', reset: false });
  } else {
    showCurrentMove();
    showButtons({ primary: null, prev: true, next: true, reset: true });
    updateStepButtons();
  }
}

// Re-scan both corners once more; readings are averaged in with prior passes.
function anotherPass() {
  state.pass = state.passes.length + 1;
  state.scanIndex = 0;
  state.faces = {};
  els.captured.innerHTML = '';
  state.phase = 'scanning';
  promptCurrentScan(
    'A few colours were ambiguous (often red vs orange). Scan once more from a ' +
    'slightly different angle — all passes are combined.'
  );
}

function failScan(message) {
  state.phase = 'error';
  els.perspectiveControl.hidden = true;
  setStatus('⚠️ Scan problem');
  setHint(`${message}<br>Tap to scan again.`);
  showButtons({ primary: 'Scan again' });
}

function updateStepButtons() {
  els.prev.disabled = state.moveIndex === 0;
  els.next.disabled = state.moveIndex >= state.solution.length;
}

function step(delta) {
  state.moveIndex = Math.max(0, Math.min(state.solution.length, state.moveIndex + delta));
  if (state.moveIndex >= state.solution.length) {
    setStatus('✓ All moves done');
    setHint('The cube should now be solved. <span class="move-chip done">restart to scan again</span>');
  } else {
    showCurrentMove();
  }
  updateStepButtons();
}

// --- events ----------------------------------------------------------------

els.primary.addEventListener('click', async () => {
  if (state.phase === 'idle') {
    try {
      setStatus('Requesting camera…');
      await startCamera(video);
      initSolver(); // warm up tables in the background
      enterScanning();
    } catch (err) {
      setStatus('⚠️ Camera error');
      setHint(err.message);
    }
  } else if (state.phase === 'scanning') {
    captureCorner();
  } else if (state.phase === 'error') {
    enterScanning();
  } else if (state.phase === 'guide') {
    enterScanning(); // "Scan another"
  }
});

// ?detect: download the current CLEAN frame (raw camera, no overlays/mirror) as a
// PNG — real test data for offline tuning. Triggerable via a visible button, a
// click on the image, or the 'c' key.
function captureFrame() {
  if (!video.videoWidth) { console.warn('[capture] no camera frame yet — start the camera first'); return; }
  const off = document.createElement('canvas');
  off.width = video.videoWidth;
  off.height = video.videoHeight;
  off.getContext('2d').drawImage(video, 0, 0);
  // Synchronous toDataURL keeps us inside the user-gesture so the download isn't
  // blocked (toBlob's async callback can lose the gesture and be silently denied).
  const a = document.createElement('a');
  a.href = off.toDataURL('image/png');
  a.download = `cube-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  document.body.appendChild(a); // Firefox needs the link in the DOM to click it
  a.click();
  a.remove();
  console.log(`[capture] saved ${a.download} (${off.width}x${off.height})`);
}
// The live-camera detect harness. Skipped under ?synth — there the synthetic frame
// is the source and the synth panel owns the page (its "Run line detector" toggle
// drives the overlay), so a second panel / camera-capture button would only conflict.
if (DEBUG_DETECT && !DEBUG_SYNTH) {
  const btn = document.createElement('button');
  btn.textContent = '📷 Save frame';
  btn.style.cssText = 'position:fixed;left:8px;bottom:84px;z-index:1000;padding:10px 14px;' +
    'font:14px system-ui;background:#111;color:#fff;border:1px solid #666;border-radius:8px;cursor:pointer';
  btn.addEventListener('click', captureFrame);
  document.body.appendChild(btn);
  canvas.addEventListener('click', captureFrame);
  window.addEventListener('keydown', (e) => { if (e.key === 'c') captureFrame(); });
  console.log('[capture] ready — use the "📷 Save frame" button (or click the image / press "c")');
  buildDetectPanel();
  // Suppress scan chrome that's irrelevant to (and overlaps) the tuning panel; the
  // mirror toggle moves into the panel.
  els.glyph.style.display = 'none';
  els.mirror.style.display = 'none';
}

if (DEBUG_SYNTH) {
  canvas.classList.remove('mirrored'); // the generated frame IS the saved image — don't flip it
  els.glyph.style.display = 'none';
  els.mirror.style.display = 'none';
  els.primary.hidden = true;
  buildSynthPanel();
  if (synthDisplay.runDetect) buildDetectPanel(); // detect controls on the right, alongside the synth panel
  console.log('[synth] generator ON; opts =', synthOpts, synthDisplay.runDetect ? '(detector overlay on)' : '');
}

// Number of decimal places implied by a slider step (2 -> 0, 0.005 -> 3). A
// function declaration (hoisted) so buildDetectPanel — called above, during module
// eval — can use it regardless of source order.
function stepDecimals(step) { const s = String(step); return s.includes('.') ? s.split('.')[1].length : 0; }

// Reflect the live tuning back into the URL so a reload (or a copied link) restores
// it — the on-page panel replaces editing the query string by hand, it doesn't
// abandon it. Only touched/overridden knobs end up in the URL, keeping it compact.
// It manages ONLY the detector's own schema keys (DETECT_PARAMS + method + hideCamera)
// and leaves everything else untouched, so under ?synth&detect it preserves the synth
// panel's params instead of wiping them. Under ?synth the synth panel owns the page,
// so the overlay is tracked by `runDetect`, not the `detect` flag.
function syncDetectURL() {
  const params = new URLSearchParams(location.search);
  for (const p of DETECT_PARAMS) params.delete(p.k);
  params.delete('method');
  params.delete('hideCamera');
  if (DEBUG_SYNTH) params.delete('detect');
  else params.set('detect', '');
  if (detectOpts.method != null) params.set('method', String(detectOpts.method));
  for (const p of DETECT_PARAMS) if (p.k in detectOpts) params.set(p.k, String(detectOpts[p.k]));
  if (!DEBUG_SYNTH && detectDisplay.hideCamera) params.set('hideCamera', '1');
  history.replaceState(null, '', location.pathname + '?' + params.toString());
}

// Build the on-page detector tuning panel (?detect only). A method <select>, the
// Mirror / Hide-camera toggles, and a slider per numeric knob for the active
// method. Sliders mutate detectOpts in place — the worker reads it every frame —
// and mirror into the URL. Reset clears overrides back to the method's defaults.
function buildDetectPanel() {
  if (document.getElementById('detect-panel')) return; // idempotent (?synth toggles it on/off)
  const panel = document.createElement('div');
  panel.id = 'detect-panel';
  panel.className = 'tune-panel';
  panel.innerHTML = `
    <h2>detect tuning</h2>
    <div class="dp-row">
      <label for="dp-method">Method</label>
      <select id="dp-method">
        <option value="canny">canny — quads</option>
        <option value="mask">mask — quads</option>
        <option value="hough">hough — lines</option>
      </select>
    </div>
    <label class="dp-toggle"><input type="checkbox" id="dp-mirror"> Mirror preview</label>
    <label class="dp-toggle"><input type="checkbox" id="dp-hide"> Hide camera (lines only)</label>
    <div id="dp-sliders"></div>
    <button id="dp-reset" type="button">Reset to defaults</button>`;
  document.body.appendChild(panel);

  const sliders = panel.querySelector('#dp-sliders');
  function renderSliders() {
    const method = currentMethod();
    const base = method === 'hough' ? { ...HOUGH_DEFAULTS, ...VP_DEFAULTS, ...ROT_DEFAULTS, ...POSE_DEFAULTS, ...POSE_SMOOTH_DEFAULTS } : DETECT_DEFAULTS;
    sliders.innerHTML = '';
    for (const p of DETECT_PARAMS) {
      if (!p.methods.includes(method)) continue;
      const dec = stepDecimals(p.step);
      const value = detectOpts[p.k] ?? base[p.k] ?? 0;
      const row = document.createElement('div');
      row.className = 'dp-slider';
      row.innerHTML = `<label><span>${p.label}</span><span class="dp-val">${Number(value).toFixed(dec)}</span></label>`;
      const input = document.createElement('input');
      input.type = 'range';
      input.min = p.min; input.max = p.max; input.step = p.step; input.value = value;
      const val = row.querySelector('.dp-val');
      input.addEventListener('input', () => {
        const v = parseFloat(input.value);
        detectOpts[p.k] = v;
        val.textContent = v.toFixed(dec);
        syncDetectURL();
      });
      row.appendChild(input);
      sliders.appendChild(row);
    }
  }

  const method = panel.querySelector('#dp-method');
  method.value = currentMethod();
  method.addEventListener('change', () => { detectOpts.method = method.value; renderSliders(); syncDetectURL(); });

  const mirror = panel.querySelector('#dp-mirror');
  mirror.checked = canvas.classList.contains('mirrored');
  mirror.addEventListener('change', () => canvas.classList.toggle('mirrored', mirror.checked));

  const hide = panel.querySelector('#dp-hide');
  hide.checked = detectDisplay.hideCamera;
  hide.addEventListener('change', () => { detectDisplay.hideCamera = hide.checked; syncDetectURL(); });

  // Under ?synth there's no camera, so the Mirror / Hide-camera toggles are meaningless.
  if (DEBUG_SYNTH) for (const id of ['#dp-mirror', '#dp-hide']) panel.querySelector(id).closest('.dp-toggle').style.display = 'none';

  panel.querySelector('#dp-reset').addEventListener('click', () => {
    for (const p of DETECT_PARAMS) delete detectOpts[p.k];
    renderSliders();
    syncDetectURL();
  });

  renderSliders();
}

// Reflect the live synth tuning back into the URL (so reload / a copied link restores
// it), preserving any non-synth params already there — e.g. detector knobs when the
// overlay is on. Only overridden knobs land in the URL, keeping it compact.
function syncSynthURL() {
  const params = new URLSearchParams(location.search);
  for (const p of SYNTH_PARAMS) params.delete(p.k);
  params.set('synth', '');
  // The overlay is canonically tracked by `runDetect`; drop the `detect` alias so the
  // two flags don't ping-pong (a fresh ?synth&detect link still enables it on load).
  params.delete('detect');
  for (const p of SYNTH_PARAMS) if (p.k in synthOpts) params.set(p.k, String(synthOpts[p.k]));
  if (synthDisplay.runDetect) params.set('runDetect', '1'); else params.delete('runDetect');
  history.replaceState(null, '', location.pathname + '?' + params.toString());
}

// Build the on-page ?synth panel: a slider per pose/appearance knob, a Run-detector
// toggle, and Save (PNG + truth JSON) / Reset buttons. Sliders mutate synthOpts in
// place — renderSynth reads it every frame — and mirror into the URL.
function buildSynthPanel() {
  const panel = document.createElement('div');
  panel.id = 'synth-panel';
  panel.className = 'tune-panel';
  panel.innerHTML = `
    <h2>synth generator</h2>
    <label class="dp-toggle"><input type="checkbox" id="sp-detect"> Run line detector</label>
    <div id="sp-sliders"></div>
    <div class="sp-actions">
      <button id="sp-save" type="button">💾 Save PNG + JSON</button>
      <button id="sp-reset" type="button">Reset to defaults</button>
    </div>`;
  document.body.appendChild(panel);

  const sliders = panel.querySelector('#sp-sliders');
  for (const p of SYNTH_PARAMS) {
    const dec = stepDecimals(p.step);
    const value = synthOpts[p.k] ?? SYNTH_DEFAULTS[p.k] ?? 0;
    const row = document.createElement('div');
    row.className = 'dp-slider';
    row.dataset.k = p.k;
    row.innerHTML = `<label><span>${p.label}</span><span class="dp-val">${Number(value).toFixed(dec)}</span></label>`;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = p.min; input.max = p.max; input.step = p.step; input.value = value;
    const val = row.querySelector('.dp-val');
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      synthOpts[p.k] = v;
      val.textContent = v.toFixed(dec);
      syncSynthURL();
    });
    row.appendChild(input);
    sliders.appendChild(row);
  }

  const detect = panel.querySelector('#sp-detect');
  detect.checked = synthDisplay.runDetect;
  detect.addEventListener('change', () => {
    synthDisplay.runDetect = detect.checked;
    if (detect.checked) { detectOpts.method ??= 'hough'; startCV(); buildDetectPanel(); } // detect controls appear on the right
    else document.getElementById('detect-panel')?.remove();
    syncSynthURL();
  });

  panel.querySelector('#sp-save').addEventListener('click', saveSynth);
  panel.querySelector('#sp-reset').addEventListener('click', () => {
    for (const p of SYNTH_PARAMS) delete synthOpts[p.k];
    for (const row of sliders.querySelectorAll('.dp-slider')) {
      const p = SYNTH_PARAMS.find((q) => q.k === row.dataset.k);
      const input = row.querySelector('input'), span = row.querySelector('.dp-val');
      input.value = SYNTH_DEFAULTS[p.k] ?? 0;
      span.textContent = Number(input.value).toFixed(stepDecimals(p.step));
    }
    syncSynthURL();
  });
}

els.next.addEventListener('click', () => step(1));
els.prev.addEventListener('click', () => step(-1));
els.reset.addEventListener('click', () => enterScanning());

els.mirror.addEventListener('click', () => {
  const on = canvas.classList.toggle('mirrored');
  els.mirror.setAttribute('aria-pressed', String(on));
});

els.perspective.addEventListener('input', () => {
  state.persp = parseFloat(els.perspective.value);
});

if (!DEBUG_SYNTH) showButtons({ primary: 'Start camera' });
requestAnimationFrame(render);

// Spin up the OpenCV worker once the first frame has painted: the ~10 MB load
// and WASM init happen off the main thread, so they're ready by scan time without
// ever freezing the UI. No user action needed. Skipped in pure ?synth (no detector
// overlay) — nothing needs OpenCV there.
if (!DEBUG_SYNTH || synthDisplay.runDetect) requestAnimationFrame(() => requestAnimationFrame(() => startCV()));
