import { startCamera } from './camera.js';
import { computeRegion, computeCornerRegion, sampleCorner, CORNER_CAPTURES } from './detection.js';
import { toFaceletString, validate } from './cube-state.js';
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
};

const state = {
  phase: 'idle', // idle | scanning | solving | guide | error
  scanIndex: 0,
  faces: {},
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
      drawCornerGuide(ctx, computeCornerRegion(canvas.width, canvas.height), true);
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
  state.faces = {};
  els.captured.innerHTML = '';
  promptCurrentScan();
}

function promptCurrentScan() {
  const cap = CORNER_CAPTURES[state.scanIndex];
  const ui = CAPTURE_UI[state.scanIndex];
  setStatus(`Scan ${state.scanIndex + 1}/${CORNER_CAPTURES.length} — <b>${cap.title}</b>`);
  setHint(ui.hint);
  setGlyph(ui.glyph);
  showButtons({ primary: 'Capture', reset: true });
}

// Show the corner-on instruction glyph ('corner'/'flip'); null hides it.
function setGlyph(motion) {
  const svg = glyphSVG(motion);
  els.glyph.innerHTML = svg;
  els.glyph.hidden = !svg;
}

function captureCorner() {
  const cap = CORNER_CAPTURES[state.scanIndex];
  const region = computeCornerRegion(canvas.width, canvas.height);
  const faces = sampleCorner(ctx, region, cap);
  for (const face of cap.faces) {
    state.faces[face.letter] = faces[face.letter];
    addThumb(faces[face.letter]);
  }

  state.scanIndex++;
  if (state.scanIndex < CORNER_CAPTURES.length) {
    promptCurrentScan();
  } else {
    solveScanned();
  }
}

async function solveScanned() {
  state.phase = 'solving';
  setStatus('Solving…');
  setHint('Computing the shortest solution.');
  setGlyph(null);
  showButtons({ primary: null, reset: true });

  const { facelets, counts } = toFaceletString(state.faces);
  const check = validate(facelets, counts);
  if (!check.ok) return failScan(check.error);

  try {
    state.solution = await solve(facelets);
  } catch (err) {
    return failScan(`Unsolvable state — a color was misread. (${err.message})`);
  }

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

function failScan(message) {
  state.phase = 'error';
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

showButtons({ primary: 'Start camera' });
requestAnimationFrame(render);
