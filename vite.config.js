import { defineConfig } from 'vite';

// cubejs is a CommonJS module; pre-bundle it so `import Cube from 'cubejs'` works.
export default defineConfig({
  server: { host: true },
  // @techstark/opencv-js is imported by the worker (src/cv-worker.js). Force it
  // into the initial dep optimize so it's pre-bundled before the worker loads;
  // otherwise the worker can trigger an on-the-fly optimize at runtime, which hangs
  // (a worker can't do the page reload Vite uses to apply newly-optimized deps).
  // The worker imports it statically, so it just loads the ready chunk.
  optimizeDeps: { include: ['cubejs', '@techstark/opencv-js'] },
  // OpenCV (~10 MB) is loaded by a module Web Worker (src/cv-worker.js) via a
  // dynamic import, so it lands in its own chunk and never bloats the app bundle.
  // Code-splitting inside a worker needs ES-module output (the default 'iife'
  // can't split). Raise the size-warning limit so that expected chunk doesn't flag.
  worker: { format: 'es' },
  build: { chunkSizeWarningLimit: 11000 },
});
