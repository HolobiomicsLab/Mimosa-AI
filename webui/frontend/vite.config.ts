import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The Observatory API runs on :8848. Proxy both REST and the live WebSocket so
// the app can use same-origin relative URLs and avoid CORS in dev.
const API_TARGET = process.env.MIMOSA_API || 'http://127.0.0.1:8848'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true, ws: true },
    },
  },
})
