import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  base: '/gust-front-detection/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/apis": "http://backend:8001", //backend to work with docker, localhost w/o
    },
  },
});
