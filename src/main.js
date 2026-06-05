import { startCamera } from './camera.js';
import { computeRegion, computeCornerRegion, sampleCorner, CORNER_CAPTURES } from './detection.js';
import { toFaceletString, validate, aggregateFaces } from './cube-state.js';
import { initSolver, solve } from './solver.js';
import { drawGuide, drawCornerGuide, drawMove } from './overlay.js';
import { glyphSVG } from './glyph.js';

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

    if (state.phase === 'scanning') {
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
  }
  requestAnimationFrame(render);
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
  const { facelets, counts } = toFaceletString(faces);

  let problem = null;
  const check = validate(facelets, counts);
  if (!check.ok) problem = check.error;

  let solution = null;
  if (!problem) {
    try {
      solution = await solve(facelets);
    } catch (err) {
      problem = `Unsolvable state — a color was misread. (${err.message})`;
    }
  }

  if (problem) {
    // Don't discard the scan — fold in another pass from a fresh angle.
    if (state.passes.length < MAX_PASSES) return anotherPass();
    return failScan(problem);
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
