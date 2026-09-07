import { defineConfig } from 'vitest/config';

// GitHub Pages serves a project site from /<repo>/. Set base at build time.
export default defineConfig({
  base: process.env.PAGES_BASE ?? '/',
  build: { outDir: 'dist', assetsDir: 'assets' },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
  },
});
