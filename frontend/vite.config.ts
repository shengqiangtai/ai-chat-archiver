import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8765',
      '/save': 'http://localhost:8765',
      '/chats': 'http://localhost:8765',
      '/search': 'http://localhost:8765',
      '/stats': 'http://localhost:8765',
      '/health': 'http://localhost:8765',
    },
  },
})
