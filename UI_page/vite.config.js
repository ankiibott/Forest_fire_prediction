// vite.config.js

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Use CommonJS style exports to reliably load PostCSS plugins 
// without complex ES Module paths.
// We are explicitly requiring the packages that you correctly installed:
// tailwindcss and autoprefixer.

export default defineConfig({
  plugins: [
    react(),
  ],
  css: {
    postcss: {
      plugins: [
        require('tailwindcss'),
        require('autoprefixer'),
      ],
    },
  },
});