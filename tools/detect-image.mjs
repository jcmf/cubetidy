// Run the real detection -> grouping -> fit pipeline on a still image (e.g. a frame
// captured from the ?detect page) and write an annotated copy. This is how we
// debug the pixel layer offline: the browser can't be scripted here, but a saved
// frame can be replayed through the exact same code with OpenCV in node, and the
// annotated PNG can be eyeballed.
//
//   node tools/detect-image.mjs frame.png [method=mask] [cannyLo=20] [minInliers=5] ...
//
// Writes frame.det.png next to each input: detected quads in green, fitted faces
// outlined in colour with solid (detected) / hollow (projection-filled) cells.

import { Canvas, loadImage } from 'skia-canvas';
import { basename, dirname, join } from 'node:path';
import { mkdirSync } from 'node:fs';
import cv from '@techstark/opencv-js';
import { detectStickerQuads } from '../src/detect.js';
import { findFaceGrids, fitFaceGrid, dedupeFaces } from '../src/group.js';
import { drawDetections, drawFittedFaces } from '../src/overlay.js';

await new Promise((r) => { if (cv && cv.Mat) return r(); cv.onRuntimeInitialized = r; });

const opts = {}, paths = [];
for (const a of process.argv.slice(2)) {
  const m = a.match(/^([A-Za-z]\w*)=(.+)$/);
  if (m) opts[m[1]] = /^-?\d*\.?\d+$/.test(m[2]) ? parseFloat(m[2]) : m[2];
  else paths.push(a);
}
if (!paths.length) {
  console.error('usage: node tools/detect-image.mjs <image> [key=val ...]');
  process.exit(1);
}

for (const p of paths) {
 try {
  const img = await loadImage(p);
  const canvas = new Canvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, img.width, img.height);

  const quads = detectStickerQuads(cv, imageData, opts);
  const faces = findFaceGrids(quads, opts);
  const fits = dedupeFaces(faces.map((f) => fitFaceGrid(f, opts)).filter(Boolean), opts);

  drawDetections(ctx, quads);
  drawFittedFaces(ctx, fits);

  const outDir = join(dirname(p), 'annotated');
  mkdirSync(outDir, { recursive: true });
  const base = basename(p).replace(/\.(png|jpe?g|webp)$/i, '');
  await canvas.saveAs(join(outDir, base + '.png'));

  // Zoomed crop around everything detected — the cube is usually small in frame.
  const pts = [...quads.flatMap((q) => q.corners), ...fits.flatMap((f) => f.outline)];
  if (pts.length) {
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y), pad = 40;
    const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad);
    const x1 = Math.min(img.width, Math.max(...xs) + pad), y1 = Math.min(img.height, Math.max(...ys) + pad);
    const cw = x1 - x0, ch = y1 - y0, scale = Math.min(3, 720 / Math.max(cw, ch));
    const zc = new Canvas(Math.round(cw * scale), Math.round(ch * scale));
    const zx = zc.getContext('2d');
    zx.drawImage(canvas, x0, y0, cw, ch, 0, 0, zc.width, zc.height);
    await zc.saveAs(join(outDir, base + '.zoom.png'));
  }

  const perFace = fits.map((f) => `${f.cells.filter((c) => c.detected).length}cells/reproj${f.reproj.toFixed(1)}px`).join(' , ');
  console.log(`${base} ${img.width}x${img.height} | method=${opts.method || 'canny'} | ` +
    `${quads.length} quads | ${faces.length} grids | ${fits.length} faces${fits.length ? ` (cells seen: ${perFace})` : ''} -> annotated/${base}.png`);
 } catch (err) {
  console.error(`${basename(p)}: FAILED — ${err.message || err}`);
 }
}
