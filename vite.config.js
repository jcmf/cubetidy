import { defineConfig } from 'vite';

// cubejs is a CommonJS module; pre-bundle it so `import Cube from 'cubejs'` works.
export default defineConfig({
  server: { host: true },
  optimizeDeps: { include: ['cubejs'] },
  // OpenCV.js (~10 MB) is code-split via a dynamic import in src/opencv.js, so it
  // lands in its own chunk and never bloats the app bundle. Raise the size-warning
  // limit so that expected, intentional chunk doesn't flag on every build.
  build: { chunkSizeWarningLimit: 11000 },
});
