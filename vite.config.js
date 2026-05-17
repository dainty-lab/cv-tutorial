import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 👇 Replace 'cv-tutorial' with YOUR GitHub repository name
export default defineConfig({
  plugins: [react()],
  base: '/cv-tutorial/',   // ← CHANGE THIS to /your-repo-name/
})
