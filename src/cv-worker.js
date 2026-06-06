// OpenCV runs entirely in this worker so the ~10 MB module load, WASM
// instantiation, and per-frame Canny/contour work never block the main thread
// (loading it on the UI thread froze the page). The main thread posts frames and
// gets sticker quads back; all cv.* lives here.
//
// detect.js is cv-injected and DOM-free, so it runs unchanged in the worker.

import { detectStickerQuads } from './detect.js';

let cv = null;

// Pull OpenCV in off the main thread; signal ready once its runtime is up.
import('@techstark/opencv-js').then((mod) => {
  const c = mod.default ?? mod;
  const ready = () => { cv = c; postMessage({ type: 'ready' }); };
  if (c && c.Mat) ready();
  else c.onRuntimeInitialized = ready;
}).catch((err) => postMessage({ type: 'error', message: String(err) }));

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'detect') return;
  if (!cv) { postMessage({ type: 'quads', quads: [] }); return; }
  // The frame's pixel buffer was transferred in; rebuild an ImageData-like object.
  const imageData = { width: msg.width, height: msg.height, data: new Uint8ClampedArray(msg.buffer) };
  let quads = [];
  try {
    quads = detectStickerQuads(cv, imageData, msg.opts);
  } catch (err) {
    postMessage({ type: 'error', message: String(err) });
  }
  postMessage({ type: 'quads', quads });
};
