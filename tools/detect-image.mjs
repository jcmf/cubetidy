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
import { basename } from 'node:path';
import cv from '@techstark/opencv-js';
import { detectStickerQuads } from '../src/detect.js';
import { findFaceGrids, fitFaceGrid } from '../src/group.js';
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
  const fits = faces.map((f) => fitFaceGrid(f, opts)).filter(Boolean);

  drawDetections(ctx, quads);
  drawFittedFaces(ctx, fits);
  const out = p.replace(/\.(png|jpe?g|webp)$/i, '') + '.det.png';
  await canvas.saveAs(out);

  const perFace = fits.map((f) => f.cells.filter((c) => c.detected).length).join('/');
  console.log(`${basename(p)} ${img.width}x${img.height} | method=${opts.method || 'canny'} | ` +
    `${quads.length} quads | ${faces.length} grids | ${fits.length} faces${fits.length ? ` (cells seen: ${perFace})` : ''} -> ${out}`);
 } catch (err) {
  console.error(`${basename(p)}: FAILED — ${err.message || err}`);
 }
}
