import { startCamera } from './camera.js';
import { computeRegion, computeCornerRegion, sampleCorner, CORNER_CAPTURES } from './detection.js';
import { toFaceletString, validate, aggregateFaces } from './cube-state.js';
import { initSolver, solve } from './solver.js';
import { drawGuide, drawCornerGuide, drawMove, drawDetections, drawCube } from './overlay.js';
import { glyphSVG } from './glyph.js';
import { startCV, cvReady, requestDetect, latestQuads } from './opencv.js';
import { findFaceGrids, fitFaceGrid } from './group.js';
import { estimateCubePose } from './cube-pose.js';
import { estimateIntrinsics } from './pose.js';

// Diagnostic harness for the OpenCV detector: open with ?detect to overlay
// detected sticker quads on the live frame while tuning against a real cube.
// Detection runs in a worker; this is pure visualization, not wired into the
// scan state machine yet.
const DEBUG_DETECT = new URLSearchParams(location.search).has('detect');
const detectState = { frame: 0, lastStatus: '', lastQuads: null, quadCount: 0, cube: null };
// Every query param besides `detect` overrides a DETECT_DEFAULTS knob, so detection
// can be tuned live from the URL without a rebuild, e.g.
//   ?detect&method=adaptive&blockSize=51&C=7   or   ?detect&cannyLo=20&minFill=0.4
const detectOpts = (() => {
  const o = {};
  for (const [k, v] of new URLSearchParams(location.search)) {
    if (k === 'detect') continue;
    o[k] = /^-?\d*\.?\d+$/.test(v) ? parseFloat(v) : v;
  }
  return o;
})();
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
  if (video.videoWidth && video.videoHeight) {
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Sample the CLEAN frame for detection here — before any guide/arrow overlay is
    // drawn onto the canvas, or the detector finds our own lines, not the cube.
    if (DEBUG_DETECT) requestDetectFrame();

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

// Ship the clean frame to the detection worker every few frames (it reads the
// whole frame). Called BEFORE overlays are drawn, so the sampled pixels are the
// raw camera image with none of our guide/arrow lines on them.
function requestDetectFrame() {
  if (!cvReady()) return;
  if (detectState.frame++ % 3 === 0) {
    requestDetect(ctx.getImageData(0, 0, canvas.width, canvas.height), detectOpts);
  }
}

// On each fresh detection result, group -> fit -> temporally smooth; draw the
// smoothed faces every frame. Count goes to the status bar, never the canvas
// (mirrored text reads backwards).
function drawDetectResults() {
  if (!cvReady()) { setDetectStatus('detect: loading OpenCV…'); return; }
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
if (DEBUG_DETECT) {
  const btn = document.createElement('button');
  btn.textContent = '📷 Save frame';
  btn.style.cssText = 'position:fixed;left:8px;bottom:84px;z-index:1000;padding:10px 14px;' +
    'font:14px system-ui;background:#111;color:#fff;border:1px solid #666;border-radius:8px;cursor:pointer';
  btn.addEventListener('click', captureFrame);
  document.body.appendChild(btn);
  canvas.addEventListener('click', captureFrame);
  window.addEventListener('keydown', (e) => { if (e.key === 'c') captureFrame(); });
  console.log('[capture] ready — use the "📷 Save frame" button (or click the image / press "c")');
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

showButtons({ primary: 'Start camera' });
requestAnimationFrame(render);

// Spin up the OpenCV worker once the first frame has painted: the ~10 MB load
// and WASM init happen off the main thread, so they're ready by scan time without
// ever freezing the UI. No user action needed.
requestAnimationFrame(() => requestAnimationFrame(() => startCV()));
