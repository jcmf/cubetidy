import { defineConfig } from 'vite';

// cubejs is a CommonJS module; pre-bundle it so `import Cube from 'cubejs'` works.
export default defineConfig({
  server: { host: true },
  // @techstark/opencv-js is only imported dynamically INSIDE the worker, which
  // Vite's startup dependency scan doesn't follow. Without forcing it here, the
  // worker's first import() triggers an on-the-fly optimize at runtime — and that
  // hangs in a worker (a worker can't do the page reload Vite uses to apply new
  // deps), so OpenCV never finishes loading. Pre-bundling it fixes the hang.
  optimizeDeps: { include: ['cubejs', '@techstark/opencv-js'] },
  // OpenCV (~10 MB) is loaded by a module Web Worker (src/cv-worker.js) via a
  // dynamic import, so it lands in its own chunk and never bloats the app bundle.
  // Code-splitting inside a worker needs ES-module output (the default 'iife'
  // can't split). Raise the size-warning limit so that expected chunk doesn't flag.
  worker: { format: 'es' },
  build: { chunkSizeWarningLimit: 11000 },
});
