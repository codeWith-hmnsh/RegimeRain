import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  // "/" for local dev, `regimerain serve` and the smoke test; "/RegimeRain/" for GitHub Pages (set in the deploy workflow)
  base: process.env.VITE_BASE ?? '/',
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: { manualChunks: (id) => (id.includes('maplibre-gl') ? 'maplibre' : undefined) },
    },
  },
})
