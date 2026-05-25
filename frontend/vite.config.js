import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";

const apiBase = (process.env.VITE_API_BASE || "/apis").replace(/\/$/, "");

// https://vite.dev/config/
export default defineConfig({
  base: '/gust-front-detection/',
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    origin: 'https://ninjastorm.firelab.org/gust-front-detection',
    proxy: {
      [apiBase]: {
        target: 'http://backend:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(apiBase, '/apis'),
      },
    },
  },
});
