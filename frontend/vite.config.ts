import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('react-markdown') || id.includes('rehype-highlight') || id.includes('highlight.js')) {
            return 'markdown-vendor'
          }
          if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
            return 'react-vendor'
          }
          return undefined
        },
      },
    },
  },
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
