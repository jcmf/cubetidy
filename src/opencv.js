// Lazy loader for OpenCV.js (the @techstark/opencv-js WASM build, ~10 MB).
//
// We never want this in the initial bundle, so it's pulled in with a dynamic
// import() — Vite emits it as a separate chunk fetched on demand. main.js calls
// loadOpenCV() after the first paint (no user action needed), and detection code
// awaits it before touching cv.*. Resolves to the initialized `cv` module once
// its WASM runtime is ready (Emscripten signals that via onRuntimeInitialized).

let readyPromise = null;
let cvRef = null;

export function loadOpenCV() {
  if (readyPromise) return readyPromise;
  readyPromise = import('@techstark/opencv-js').then((mod) => {
    const cv = mod.default ?? mod;
    return new Promise((resolve) => {
      if (cv && cv.Mat) { cvRef = cv; return resolve(cv); } // already initialized
      cv.onRuntimeInitialized = () => { cvRef = cv; resolve(cv); };
    });
  });
  return readyPromise;
}

// Synchronous "is it usable yet?" check, for UI gating without awaiting.
export function openCVReady() {
  return !!(cvRef && cvRef.Mat);
}
