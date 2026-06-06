// OpenCV runs in this module worker so the ~10 MB load, WASM init, and per-frame
// detection stay off the main thread (loading it on the UI thread froze the page
// after first render). The main thread posts frames in and gets sticker quads back.
//
// Readiness is detected by POLLING for cv.Mat, NOT by onRuntimeInitialized.
// OpenCV's runtime initializes fine here (its wasm is an embedded data URI, so no
// fetch/environment-specific loading is involved — confirmed by simulating this
// worker env), but through the dynamic import + bundler the onRuntimeInitialized
// callback proved unreliable: it can fire before this .then runs, or on a different
// object reference than the imported default. A working Mat constructor is the
// ground truth, and appears ~80 ms after the import resolves.
//
// detect.js is cv-injected and DOM-free, so it runs here unchanged.

import { detectStickerQuads } from './detect.js';

let cv = null;
console.log('[cv-worker] importing OpenCV…');
import('@techstark/opencv-js').then((mod) => {
  cv = mod.default ?? mod;
  console.log('[cv-worker] import resolved (default:', typeof cv, ', has Mat:', !!(cv && cv.Mat), ')');
  const t0 = Date.now();
  const waitReady = () => {
    if (cv && cv.Mat) { console.log('[cv-worker] OpenCV runtime ready'); postMessage({ type: 'ready' }); return; }
    if (Date.now() - t0 > 20000) { postMessage({ type: 'error', message: 'OpenCV init timed out (cv.Mat never appeared)' }); return; }
    setTimeout(waitReady, 50);
  };
  waitReady();
}).catch((err) => {
  console.error('[cv-worker] OpenCV load failed', err);
  postMessage({ type: 'error', message: String(err) });
});

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
