import { defineConfig, devices } from "@playwright/test";

// E2E runs against the static build (VITE_STATIC=1 npm run build && ctxprof export --out web/dist),
// which is exactly what GitHub Pages serves. Set E2E_BASE_URL to test a live `ctxprof serve` instead.
const external = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: external ?? "http://127.0.0.1:4173/", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: external ? undefined : {
    command: "npx vite preview --port 4173 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4173/",
    reuseExistingServer: !process.env.CI,
  },
});
