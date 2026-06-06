// OpenCV runs in this module worker so the ~10 MB load, WASM init, and per-frame
// detection stay off the main thread (loading it on the UI thread froze the page
// after first render). The main thread posts frames in and gets sticker quads back.
//
// OpenCV is imported STATICALLY (not via a dynamic import()). A dynamic import of a
// bare dep inside a worker hangs in Vite dev: that dep isn't followed by the
// startup dependency scan, so the first import() triggers an on-the-fly optimize at
// runtime — which never completes in a worker (a worker can't do the page reload
// Vite uses to apply newly-optimized deps). A static import IS followed by the
// scan, so OpenCV is pre-bundled before the worker runs and just loads normally.
//
// Readiness is detected by POLLING for cv.Mat, NOT onRuntimeInitialized: through
// the bundler that callback fires unreliably (before our code runs, or on a
// different object). A working Mat constructor is the ground truth; in a faithful
// node simulation of this worker env it appears ~80 ms after the module loads.
//
// detect.js is cv-injected and DOM-free, so it runs here unchanged.

import { detectStickerQuads } from './detect.js';
import cvModule from '@techstark/opencv-js';

const cv = cvModule?.default ?? cvModule; // tolerate either interop shape
let ready = false;

console.log('[cv-worker] OpenCV module imported; waiting for runtime…');
const t0 = Date.now();
const waitReady = () => {
  if (cv && cv.Mat) { ready = true; console.log('[cv-worker] OpenCV runtime ready'); postMessage({ type: 'ready' }); return; }
  if (Date.now() - t0 > 20000) { postMessage({ type: 'error', message: 'OpenCV init timed out (cv.Mat never appeared)' }); return; }
  setTimeout(waitReady, 50);
};
waitReady();

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'detect') return;
  if (!ready) { postMessage({ type: 'quads', quads: [] }); return; }
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
