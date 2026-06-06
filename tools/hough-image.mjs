// Run the Canny + probabilistic-Hough line explorer (src/detect.js
// detectLineSegments) on a still image and write an annotated copy. Mirrors
// tools/detect-image.mjs but for the line-segment path: the browser can't be
// scripted here, so a saved ?detect frame is replayed through the exact same code
// with OpenCV in node and the annotated PNG eyeballed. Also proves the HoughLinesP
// cv.* calls / Mat accessors / cleanup are correct (the pixel layer is the only
// part `npm test` can't cover).
//
//   node tools/hough-image.mjs frame.png [cannyLo=40] [houghThresh=50] [minLineLen=30] ...
//
// Writes annotated/<name>.hough.png next to each input (segments hue-coded by
// orientation, matching the live overlay). Pass bg=black to draw the segments on
// black instead of over the frame — the clearest way to judge line quality when
// the cube itself is brightly coloured.

import { Canvas, loadImage } from 'skia-canvas';
import { basename, dirname, join } from 'node:path';
import { mkdirSync } from 'node:fs';
import cv from '@techstark/opencv-js';
import { detectLineSegments } from '../src/detect.js';
import { drawSegments } from '../src/overlay.js';

await new Promise((r) => { if (cv && cv.Mat) return r(); cv.onRuntimeInitialized = r; });

const opts = {}, paths = [];
for (const a of process.argv.slice(2)) {
  const m = a.match(/^([A-Za-z]\w*)=(.+)$/);
  if (m) opts[m[1]] = /^-?\d*\.?\d+$/.test(m[2]) ? parseFloat(m[2]) : m[2];
  else paths.push(a);
}
if (!paths.length) {
  console.error('usage: node tools/hough-image.mjs <image> [key=val ...]');
  process.exit(1);
}

for (const p of paths) {
 try {
  const img = await loadImage(p);
  const canvas = new Canvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, img.width, img.height);

  const segments = detectLineSegments(cv, imageData, opts);
  if (opts.bg === 'black') { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, img.width, img.height); }
  // Thicker than the live 2px so segments stay visible when the frame is downscaled.
  drawSegments(ctx, segments, 3);

  const outDir = join(dirname(p), 'annotated');
  mkdirSync(outDir, { recursive: true });
  const base = basename(p).replace(/\.(png|jpe?g|webp)$/i, '');
  await canvas.saveAs(join(outDir, base + '.hough.png'));

  // Zoomed crop around the segments — the cube is usually small in frame and thin
  // 2px lines vanish when the full frame is downscaled for viewing.
  const pts = segments.flatMap((s) => [{ x: s.x1, y: s.y1 }, { x: s.x2, y: s.y2 }]);
  if (pts.length) {
    const xs = pts.map((q) => q.x), ys = pts.map((q) => q.y), pad = 40;
    const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad);
    const x1 = Math.min(img.width, Math.max(...xs) + pad), y1 = Math.min(img.height, Math.max(...ys) + pad);
    const cw = x1 - x0, ch = y1 - y0, scale = Math.min(3, 720 / Math.max(cw, ch));
    const zc = new Canvas(Math.round(cw * scale), Math.round(ch * scale));
    zc.getContext('2d').drawImage(canvas, x0, y0, cw, ch, 0, 0, zc.width, zc.height);
    await zc.saveAs(join(outDir, base + '.hough.zoom.png'));
  }

  console.log(`${base} ${img.width}x${img.height} | ${segments.length} segments -> annotated/${base}.hough.png`);
 } catch (err) {
  console.error(`${basename(p)}: FAILED — ${err.message || err}`);
 }
}
