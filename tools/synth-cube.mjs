// Generate SYNTHETIC cube frames with KNOWN ground truth — a controlled cube at a
// chosen 6-DoF pose, rendered to a PNG, with a sidecar JSON recording exactly what
// was drawn (pose, intrinsics, per-sticker colours, projected corners/cells).
//
// Why this exists: the line-detector splits into an offline-testable GEOMETRY half
// (groupLineSegments → solveCubeFromLines → recoverCubePose, round-tripped in
// test/lines.test.mjs from pre-projected segments) and a PIXEL half (Canny + Hough
// in src/detect.js) that until now could only be run on the real captured frames in
// samples/ — which have NO ground truth, so you can eyeball the wireframe but can't
// measure pose error. A synthetic frame closes that: it has a known pose, so the
// REAL pixel pipeline can be pushed end-to-end and the recovered pose checked
// against truth (tools/synth-smoke.mjs does exactly that).
//
// The scene/draw core lives in src/synth.js (browser-safe, shared with the in-page
// ?synth harness); this file only adds the node bits — a skia canvas to draw onto,
// and file I/O for the PNG + ground-truth JSON.
//
//   node tools/synth-cube.mjs [out.png] [key=val ...]
//   node tools/synth-cube.mjs samples/synth/a.png angleDeg=57 dist=5 axis=0.9,-1,0.1
//
// Knobs: width,height,fovDeg | axis=x,y,z angleDeg dist tx ty (pose) |
//        scramble=<seed> (random per-sticker colours; 0=solved) |
//        gap bg imgBlur(=blur) noise (rendering). Writes <out>.png and <out>.truth.json.

import { Canvas } from 'skia-canvas';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCubeScene, drawScene } from '../src/synth.js';

// Convenience re-export so callers (e.g. tools/synth-smoke.mjs) get both from here.
export { buildCubeScene };

// Node-side render: build a skia canvas the scene's size and draw onto it.
export function renderScene(scene, opts = {}) {
  const canvas = new Canvas(scene.width, scene.height);
  drawScene(canvas.getContext('2d'), scene, opts);
  return canvas;
}

// --- CLI --------------------------------------------------------------------

function parseArgs(argv) {
  const opts = {}, paths = [];
  for (const a of argv) {
    const m = a.match(/^([A-Za-z]\w*)=(.+)$/);
    if (m) opts[m[1]] = /^-?\d*\.?\d+$/.test(m[2]) ? parseFloat(m[2]) : m[2];
    else paths.push(a);
  }
  return { opts, paths };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const { opts, paths } = parseArgs(process.argv.slice(2));
  const out = paths[0] || 'samples/synth/cube.png';
  const scene = buildCubeScene(opts);
  const canvas = renderScene(scene, opts);

  mkdirSync(dirname(out), { recursive: true });
  await canvas.toFile(out);
  const truthPath = out.replace(/\.(png|jpe?g|webp)$/i, '') + '.truth.json';
  writeFileSync(truthPath, JSON.stringify(scene.truth, null, 2));

  const tr = scene.truth;
  console.log(`${basename(out)} ${scene.width}x${scene.height} | faces ${tr.visibleFaces.map((f) => f.letter).join('')} | `
    + `angle ${tr.angleDeg == null ? 'R' : tr.angleDeg.toFixed(0) + '°'} dist ${tr.dist} | edge ${tr.edgePx.toFixed(0)}px -> ${out} (+ ${basename(truthPath)})`);
}
