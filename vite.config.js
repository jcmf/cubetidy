import { defineConfig } from 'vite';

// cubejs is a CommonJS module; pre-bundle it so `import Cube from 'cubejs'` works.
export default defineConfig({
  server: { host: true },
  optimizeDeps: { include: ['cubejs'] },
});
