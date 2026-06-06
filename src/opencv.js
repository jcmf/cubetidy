// Main-thread handle to the OpenCV worker (src/cv-worker.js). All of OpenCV —
// the ~10 MB load, WASM init, and per-frame detection — runs in that worker, so
// the UI thread stays responsive (loading it inline froze the page after render).
//
// startCV() spins the worker up after the first paint; requestDetect() ships a
// frame over (transferring its pixel buffer, zero-copy) but only when the worker
// is idle, so requests never pile up; the newest quads are cached for drawing.

let worker = null;
let ready = false;
let inFlight = false;
let latest = [];

export function startCV() {
  if (worker) return;
  worker = new Worker(new URL('./cv-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') ready = true;
    else if (msg.type === 'quads') { inFlight = false; latest = msg.quads; }
    else if (msg.type === 'error') { inFlight = false; console.warn('cv-worker:', msg.message); }
  };
  worker.onerror = (e) => console.warn('cv-worker error', e.message || e);
}

// True once OpenCV's runtime is up in the worker.
export function cvReady() { return ready; }

// Ask the worker to detect on this frame, unless it's still busy with the last
// one. Transfers imageData's buffer — pass a fresh getImageData result each time.
export function requestDetect(imageData, opts) {
  if (!ready || inFlight) return;
  inFlight = true;
  const buffer = imageData.data.buffer;
  worker.postMessage(
    { type: 'detect', width: imageData.width, height: imageData.height, buffer, opts },
    [buffer],
  );
}

// The most recent quads returned by the worker (drawn every frame).
export function latestQuads() { return latest; }
