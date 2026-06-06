// API-level smoke test for src/detect.js against the real OpenCV.js build, run in
// node. It synthesizes a frame of solid squares on a plain background and checks
// detectStickerQuads finds them without throwing or leaking. This proves the cv.*
// calls, constants, Mat accessors and cleanup are correct — NOT that detection is
// robust on a real cube (that needs a browser + camera). Not part of `npm test`
// (loads the 10 MB WASM); run manually:  node tools/detect-smoke.mjs

import cv from '@techstark/opencv-js';
import { detectStickerQuads } from '../src/detect.js';

await new Promise((res) => {
  if (cv && cv.Mat) return res();
  cv.onRuntimeInitialized = res;
});

// Build a fake frame: gray background with a 3x3 grid of separated solid squares.
const W = 480, H = 480;
const data = new Uint8ClampedArray(W * H * 4);
for (let i = 0; i < W * H; i++) {
  data[i * 4] = 150; data[i * 4 + 1] = 150; data[i * 4 + 2] = 150; data[i * 4 + 3] = 255;
}
const fillRect = (x0, y0, w, h, [r, g, b]) => {
  for (let y = y0; y < y0 + h; y++)
    for (let x = x0; x < x0 + w; x++) {
      const i = (y * W + x) * 4;
      data[i] = r; data[i + 1] = g; data[i + 2] = b;
    }
};
const COLORS = [[220, 40, 40], [40, 200, 40], [40, 40, 220], [230, 230, 40],
  [240, 130, 30], [240, 240, 240], [200, 30, 120], [30, 200, 200], [120, 60, 200]];
const sq = 70, gap = 50, ox = 55, oy = 55;
let k = 0;
for (let r = 0; r < 3; r++)
  for (let c = 0; c < 3; c++)
    fillRect(ox + c * (sq + gap), oy + r * (sq + gap), sq, sq, COLORS[k++]);

const imageData = { width: W, height: H, data };

let fail = 0;
const check = (name, cond, extra = '') => {
  if (!cond) fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`));
};

const quads = detectStickerQuads(cv, imageData);
check('returns an array', Array.isArray(quads));
check('finds the 9 squares (>=9 quads)', quads.length >= 9, `got ${quads.length}`);
check('every quad has 4 corners + a center', quads.every((q) =>
  q.corners?.length === 4 && Number.isFinite(q.center?.x) && Number.isFinite(q.center?.y)));
check('quad centers land inside the frame', quads.every((q) =>
  q.center.x >= 0 && q.center.x < W && q.center.y >= 0 && q.center.y < H));

// Run a few more times to surface any Mat leak/double-free crash.
let crashed = false;
try { for (let i = 0; i < 10; i++) detectStickerQuads(cv, imageData); } catch (e) { crashed = true; console.log(e); }
check('repeated calls do not crash (memory handling)', !crashed);

console.log(`\n${fail ? 'FAILED' : 'OK'} — ${fail} failure(s); detected ${quads.length} quads`);
process.exit(fail ? 1 : 0);
