import { defineConfig } from 'vite';

// cubejs is a CommonJS module; pre-bundle it so `import Cube from 'cubejs'` works.
export default defineConfig({
  server: { host: true },
  optimizeDeps: { include: ['cubejs'] },
  // OpenCV runs in a Web Worker (src/cv-worker.js) that dynamically imports the
  // ~10 MB module. Code-splitting inside a worker needs ES-module output (the
  // default 'iife' can't split). The OpenCV chunk lands separately, so raise the
  // size-warning limit so that expected, intentional chunk doesn't flag.
  worker: { format: 'es' },
  build: { chunkSizeWarningLimit: 11000 },
});
