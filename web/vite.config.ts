import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies /api to `ctxprof serve` on :8000.
export default defineConfig({
  plugins: [react()],
  base: "./", // relative assets: works at / (ctxprof serve) and under a sub-path (GitHub Pages)
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  preview: { proxy: {} }, // the static build reads ./api/*.json; never proxy it
  build: { outDir: "dist", emptyOutDir: true },
});
