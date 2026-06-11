// Synthetic BENCHMARK for the line-based cube detector — generates a deterministic
// matrix of ground-truth frames (views × distances × appearance, with seeded pose
// jitter), runs the REAL pixel pipeline (detectLineSegments → solveCubeFromLines) on
// each, and grades the recovered pose against truth. Where tools/synth-smoke.mjs is a
// small hand-picked PASS/FAIL gate, this is a MEASURING instrument: it reports lock /
// accuracy / false-lock rates and error medians per tier, writes a JSON result, and
// can diff against a saved baseline so a detector change shows exactly which scenes
// it won or lost.
//
//   node tools/synth-bench.mjs [key=val ...]        (npm run synth:bench)
//
// Bench knobs:
//   out=samples/synth/bench.json  where the JSON result is written
//   baseline=<path>   diff a previous result; exits 1 on per-scene regressions
//                     (typical flow: run once, `cp bench.json bench-base.json`,
//                     change the detector, run with baseline=samples/synth/bench-base.json)
//   filter=<substr>   only scenes whose id contains the substring
//   seed=1            pose-jitter seed. Changing it resamples every pose — a cheap
//                     robustness check — but baselines only diff against the same seed.
//   dump=1            write PNG + truth JSON for each problem scene to
//                     samples/synth/bench/ (ready for tools/hough-image.mjs)
//   verbose=1         print every scene line, not just problem scenes
// Any OTHER key=val is passed through to the detector AND solver (houghThresh=40,
// minCover=0.5, vpMaxErrorDeg=4, ...), so a knob's aggregate effect across the whole
// matrix is measured in one run. (Comma values become arrays, so vpSweep=3,4,5 works.)
//
// Tiers and what "good" means:
//   core  3 visible faces, dist 3–6 (the detector's claimed corner-on regime):
//         expect an ACCURATE lock.
//   edge  2 visible faces, dist 3–6: expect an accurate lock too (the cell-hop
//         translation fix covers these). core/edge are derived from each scene's TRUE
//         visible-face count, not the view name — pose jitter can take a nominally
//         3-face view to 2 faces and vice versa.
//   far   dist 9, outside the claimed regime (near-affine, weak VPs): a MISS is fine
//         and expected; what is graded is FALSE LOCKS — a confident wrong overlay is
//         the one truly bad outcome out there.
//
// Per-scene grading (same tolerances as synth-smoke): locked; rotation error mod the
// 24 cube symmetries ≤ 8°; projected centre ≤ 0.25·edge; depth ≤ 18%. Plus a stricter
// end-task metric, CELL HIT RATE: the fraction of visible sticker centres that
// reproject — under the recovered pose, symmetry-aligned to truth — within half a
// cell of where they really are, i.e. "would the colour reader sample the right
// sticker". outcome ∈ accurate | falseLock (locked but out of tolerance) | miss.
//
// Rendering is byte-deterministic (see drawScene in src/synth.js), so the same seed
// gives identical pixels, segments and grades on every run — any delta IS the code
// change. Loads the OpenCV WASM like detect-smoke/synth-smoke, so NOT part of
// `npm test`; run it manually around detector work.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import cv from '@techstark/opencv-js';
import { buildCubeScene, renderScene } from './synth-cube.mjs';
import { detectLineSegments } from '../src/detect.js';
import { solveCubeFromLines, canonicalizeRotation, rotationAngleDeg } from '../src/lines.js';
import { project } from '../src/pose.js';

const ANGLE_TOL = 8;     // recovered R within this many degrees of truth (mod symmetry)
const CENTRE_TOL = 0.25; // projected centre within this fraction of the cube edge
const DEPTH_TOL = 0.18;  // recovered depth within this fraction of true depth

// --- the scene matrix --------------------------------------------------------
// Views span the claimed regime (3-face corner-on/tilted, separated VPs) plus the
// 2-face near-edge-on case; dist 9 is deliberately OUT of regime (far tier). maxOff
// bounds |tx|,|ty| (cube-edge units) so the cube stays mostly in frame — tight at
// dist 3, where the cube nearly fills the frame height and clipping its edge is part
// of the regime being tested.
const VIEWS = [
  { name: 'cornerA', axis: [0.9, -1, 0.1], angleDeg: 57 },
  { name: 'cornerB', axis: [0.6, -1, 0.2], angleDeg: 60 },
  { name: 'cornerC', axis: [-0.55, -1, 0.1], angleDeg: 57 },
  { name: 'tilted', axis: [0.4, -1, 0.5], angleDeg: 65 },
  { name: 'edgeOn', axis: [0.15, -1, 0.1], angleDeg: 57 }, // typically 2-face → edge tier
];
const DISTS = [
  { name: 'd3', dist: 3, maxOff: 0.15 },
  { name: 'd4.5', dist: 4.5, maxOff: 0.35 },
  { name: 'd6', dist: 6, maxOff: 0.6 },
  { name: 'd9', dist: 9, maxOff: 0.9, tier: 'far' },
];
const LOOKS = [
  { name: 'clean', opts: {} },
  { name: 'blur', opts: { imgBlur: 2 } },
  { name: 'noise', opts: { noise: 10 } },
  { name: 'soft', opts: { imgBlur: 2, noise: 8 } },          // blur + noise ≈ a mediocre webcam
  { name: 'scramble', opts: { scramble: 1, imgBlur: 1, noise: 5 } }, // per-sticker colours
];

// Same generator as src/synth.js (not exported there); only used for pose jitter.
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Build the deterministic suite: every (view, dist, look) cell, each with a seeded
// jitter of axis/angle/offset/dist so the matrix samples a neighbourhood rather than
// 20 exact hand-picked poses. Params are ROUNDED first and the rounded values used to
// build the scene, so the repro command in the report regenerates the byte-identical
// frame.
function buildSuite(seed) {
  const r2 = (v) => Math.round(v * 100) / 100;
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const scenes = [];
  let i = 0;
  for (const view of VIEWS) for (const d of DISTS) for (const look of LOOKS) {
    const rand = rng(seed * 100003 + i * 257 + 1);
    const jit = () => rand() - 0.5;
    const params = {
      axis: view.axis.map((a) => r3(a + jit() * 0.12)).join(','),
      angleDeg: r2(view.angleDeg + jit() * 8),
      dist: r2(d.dist * (1 + jit() * 0.12)),
      tx: r3(jit() * 2 * d.maxOff),
      ty: r3(jit() * 2 * d.maxOff),
      ...look.opts,
    };
    if (look.opts.scramble) params.scramble = 1 + i; // vary the colour pattern per scene
    if (look.opts.noise) params.seed = 100 + i;      // vary the noise field per scene
    scenes.push({ id: `${view.name}-${d.name}-${look.name}`, tier: d.tier ?? null, params });
    i++;
  }
  return scenes;
}

// --- grading ------------------------------------------------------------------

function gradeScene(spec, detOpts) {
  const scene = buildCubeScene(spec.params);
  const canvas = renderScene(scene, spec.params);
  const imageData = canvas.getContext('2d').getImageData(0, 0, scene.width, scene.height);

  const t0 = performance.now();
  const segments = detectLineSegments(cv, imageData, detOpts);
  const t1 = performance.now();
  const sol = solveCubeFromLines(segments, scene.K, detOpts);
  const t2 = performance.now();

  const fit = sol && sol.fit;
  // Tier from the scene's TRUE geometry: far is distance-based, otherwise the actual
  // visible-face count decides (jitter can flip a view between 3-face and 2-face).
  const tier = spec.tier ?? (scene.truth.visibleFaces.length <= 2 ? 'edge' : 'core');
  const rec = {
    id: spec.id, tier, params: spec.params,
    faces: scene.truth.visibleFaces.map((f) => f.letter).join(''),
    segments: segments.length, detectMs: Math.round(t1 - t0), solveMs: Math.round(t2 - t1),
    locked: !!(fit && fit.locked), accurate: false,
  };
  if (fit) {
    rec.cover = +(fit.cover ?? 0).toFixed(3);
    rec.reprojErr = Number.isFinite(fit.reprojErr) ? +fit.reprojErr.toFixed(2) : null;
  }
  if (rec.locked) {
    // Rotation mod the cube's 24 symmetries: align to the rep nearest truth, then the
    // residual angle is the real orientation error (and the alignment is the cell
    // correspondence the colour reader would recover).
    const Rc = canonicalizeRotation(fit.pose.R, scene.pose.R);
    rec.rotErrDeg = +rotationAngleDeg(scene.pose.R, Rc).toFixed(2);

    const cT = project(scene.K, scene.pose, [0, 0, 0]);
    const cF = project(scene.K, fit.pose, [0, 0, 0]);
    rec.centreErrEdge = +(Math.hypot(cF[0] - cT[0], cF[1] - cT[1]) / scene.truth.edgePx).toFixed(3);
    rec.depthErrFrac = +(Math.abs(fit.pose.t[2] - scene.pose.t[2]) / scene.pose.t[2]).toFixed(3);

    // Cell hit rate — the end-task metric: would each visible sticker centre,
    // reprojected under the recovered (symmetry-aligned) pose, still land inside its
    // own cell (within half a cell of truth)?
    const poseC = { R: Rc, t: fit.pose.t };
    const cellPx = scene.truth.edgePx / 3;
    let hits = 0;
    for (const s of scene.stickers) {
      const c3 = [0, 1, 2].map((a) => (s.quad3D[0][a] + s.quad3D[1][a] + s.quad3D[2][a] + s.quad3D[3][a]) / 4);
      const pT = project(scene.K, scene.pose, c3);
      const pF = project(scene.K, poseC, c3);
      if (Math.hypot(pF[0] - pT[0], pF[1] - pT[1]) < cellPx / 2) hits++;
    }
    rec.cellHitRate = +(hits / scene.stickers.length).toFixed(3);

    rec.accurate = rec.rotErrDeg <= ANGLE_TOL && rec.centreErrEdge <= CENTRE_TOL && rec.depthErrFrac <= DEPTH_TOL;
  }
  rec.outcome = !rec.locked ? 'miss' : rec.accurate ? 'accurate' : 'falseLock';
  return { rec, canvas, truth: scene.truth };
}

// A problem worth a human look: core/edge anything short of accurate; far only a
// false lock (misses out there are the expected, correct behaviour).
const isProblem = (r) => (r.tier === 'far' ? r.outcome === 'falseLock' : r.outcome !== 'accurate');

// --- reporting ----------------------------------------------------------------

const median = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const fmt = (v, d = 1) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d));
const pad = (s, w) => String(s).padStart(w);

function summarize(records) {
  const tiers = {};
  for (const r of records) {
    const t = (tiers[r.tier] ??= { n: 0, locked: 0, accurate: 0, falseLocks: 0, rot: [], ctr: [], dep: [], cell: [], ms: [] });
    t.n++;
    t.ms.push(r.detectMs + r.solveMs);
    if (r.locked) {
      t.locked++;
      r.accurate ? t.accurate++ : t.falseLocks++;
      t.rot.push(r.rotErrDeg); t.ctr.push(r.centreErrEdge); t.dep.push(r.depthErrFrac); t.cell.push(r.cellHitRate);
    }
  }
  return tiers;
}

const tierStats = (t) => ({
  n: t.n, locked: t.locked, accurate: t.accurate, falseLocks: t.falseLocks,
  medRotErrDeg: median(t.rot), medCentreErrEdge: median(t.ctr), medDepthErrFrac: median(t.dep),
  medCellHitRate: median(t.cell), medMs: median(t.ms),
});

function sceneLine(r) {
  const tag = { accurate: '  ok ', miss: ' MISS', falseLock: 'FALSE' }[r.outcome];
  let s = `${tag}  ${r.id.padEnd(24)} ${r.tier.padEnd(4)} ${pad(r.segments, 3)}seg`;
  if (r.locked) {
    s += `  rot ${fmt(r.rotErrDeg)}°  ctr ${fmt(r.centreErrEdge, 2)}·e  dep ${fmt(r.depthErrFrac * 100, 0)}%`
      + `  cell ${fmt(r.cellHitRate * 100, 0)}%  cover ${fmt(r.cover, 2)}`;
  }
  return s;
}

const reproCmd = (r) =>
  `node tools/synth-cube.mjs samples/synth/bench/${r.id}.png `
  + Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join(' ');

// Per-scene diff vs a saved result. Rank: accurate > miss > falseLock (a wrong
// confident overlay is worse than no lock). On the far tier only falseLock
// transitions matter; accurate↔miss flips out there are reported but neutral.
function diffBaseline(records, base) {
  const byId = new Map(base.scenes.map((r) => [r.id, r]));
  const rank = { falseLock: 0, miss: 1, accurate: 2 };
  const regressions = [], improvements = [], neutral = [];
  for (const r of records) {
    const b = byId.get(r.id);
    if (!b || b.outcome === r.outcome) continue;
    const line = `${r.id} (${r.tier}): ${b.outcome} -> ${r.outcome}`;
    if (r.tier === 'far') {
      if (r.outcome === 'falseLock') regressions.push(line);
      else if (b.outcome === 'falseLock') improvements.push(line);
      else neutral.push(line);
    } else (rank[r.outcome] < rank[b.outcome] ? regressions : improvements).push(line);
  }
  return { regressions, improvements, neutral };
}

// --- main ----------------------------------------------------------------------

function parseArgs(argv) {
  const opts = {};
  for (const a of argv) {
    const m = a.match(/^([A-Za-z]\w*)=(.+)$/);
    if (m) opts[m[1]] = /^-?\d*\.?\d+$/.test(m[2]) ? parseFloat(m[2]) : m[2];
  }
  return opts;
}

await new Promise((r) => { if (cv && cv.Mat) return r(); cv.onRuntimeInitialized = r; });

const opts = parseArgs(process.argv.slice(2));
const BENCH_KEYS = new Set(['out', 'baseline', 'filter', 'seed', 'dump', 'verbose']);
const detOpts = {};
for (const [k, v] of Object.entries(opts)) {
  if (BENCH_KEYS.has(k)) continue;
  // "3,4,5" -> [3,4,5] so array knobs like vpSweep are passable from the CLI.
  detOpts[k] = typeof v === 'string' && /^-?[\d.]+(,-?[\d.]+)+$/.test(v) ? v.split(',').map(Number) : v;
}
const seed = +opts.seed || 1;
const outPath = opts.out || 'samples/synth/bench.json';

let suite = buildSuite(seed);
if (opts.filter) suite = suite.filter((s) => s.id.includes(opts.filter));
console.log(`synth-bench: ${suite.length} scenes, seed ${seed}`
  + (opts.filter ? `, filter "${opts.filter}"` : '')
  + (Object.keys(detOpts).length ? `, detector overrides ${JSON.stringify(detOpts)}` : ''));

const records = [];
for (const spec of suite) {
  const { rec, canvas, truth } = gradeScene(spec, detOpts);
  records.push(rec);
  if (opts.verbose) console.log(sceneLine(rec));
  else process.stdout.write(rec.outcome === 'accurate' ? '.' : isProblem(rec) ? 'F' : 'm');
  if (opts.dump && isProblem(rec)) {
    mkdirSync('samples/synth/bench', { recursive: true });
    await canvas.toFile(`samples/synth/bench/${rec.id}.png`);
    writeFileSync(`samples/synth/bench/${rec.id}.truth.json`, JSON.stringify(truth, null, 2));
  }
}
if (!opts.verbose) console.log();

// Summary table: counts per tier, then medians over the LOCKED scenes of that tier.
const tiers = summarize(records);
const order = ['core', 'edge', 'far'].filter((k) => tiers[k]);
console.log('\ntier      n  lock   acc  false |  rot°  ctr·e  dep%  cell% |   ms');
for (const name of order) {
  const t = tiers[name];
  console.log(`${name.padEnd(6)} ${pad(t.n, 4)} ${pad(t.locked, 5)} ${pad(t.accurate, 5)} ${pad(t.falseLocks, 6)}`
    + ` | ${pad(fmt(median(t.rot)), 5)} ${pad(fmt(median(t.ctr), 2), 6)} ${pad(fmt(median(t.dep) * 100, 0), 5)}`
    + ` ${pad(fmt(median(t.cell) * 100, 0), 6)} | ${pad(fmt(median(t.ms), 0), 4)}`);
}

const problems = records.filter(isProblem);
if (problems.length) {
  console.log(`\n${problems.length} problem scene(s) (core/edge below accurate; far false locks):`);
  for (const r of problems) {
    console.log(sceneLine(r));
    console.log(`       repro: ${reproCmd(r)}`);
  }
}

mkdirSync('samples/synth', { recursive: true });
writeFileSync(outPath, JSON.stringify({
  date: new Date().toISOString(), seed, filter: opts.filter || null, detectorOpts: detOpts,
  tolerances: { ANGLE_TOL, CENTRE_TOL, DEPTH_TOL },
  summary: Object.fromEntries(order.map((k) => [k, tierStats(tiers[k])])),
  scenes: records,
}, null, 1));
console.log(`\nwrote ${outPath} (${records.length} scenes)`);

let exitCode = 0;
if (opts.baseline) {
  const base = JSON.parse(readFileSync(opts.baseline, 'utf8'));
  if (base.seed !== seed) {
    console.log(`\nbaseline ${opts.baseline} used seed ${base.seed}, this run ${seed} — poses differ, per-scene diff skipped`);
  } else {
    const { regressions, improvements, neutral } = diffBaseline(records, base);
    console.log(`\nvs baseline ${opts.baseline} (${base.date}):`);
    for (const l of regressions) console.log(`  REGRESSION  ${l}`);
    for (const l of improvements) console.log(`  improved    ${l}`);
    for (const l of neutral) console.log(`  (far flip)  ${l}`);
    if (!regressions.length && !improvements.length && !neutral.length) console.log('  no per-scene outcome changes');
    exitCode = regressions.length ? 1 : 0;
  }
}
process.exit(exitCode);
