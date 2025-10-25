// vite.config.js

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react(),
  ],
  css: {
    postcss: {
      plugins: [
        // CRITICAL FIX: Use the new dedicated PostCSS plugin package
        require('@tailwindcss/postcss'),
        // Keep autoprefixer
        require('autoprefixer'),
      ],
    },
  },
});