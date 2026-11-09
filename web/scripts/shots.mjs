// Screenshot review helper: node scripts/shots.mjs [baseUrl] [outDir]
// Captures every screen in dark, light and at 375 px, and reports horizontal overflow.
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:8000/";
const out = process.argv[3] ?? "shots";
mkdirSync(out, { recursive: true });
const pages = { overview: "#/", segments: "#/segments", pruning: "#/pruning", trace: "#/trace/t0014", held: "#/trace/t0213" };
const modes = [
  { name: "dark", width: 1440, height: 1000, scheme: "dark" },
  { name: "light", width: 1440, height: 1000, scheme: "light" },
  { name: "phone", width: 375, height: 812, scheme: "dark" },
];

const browser = await chromium.launch();
for (const m of modes) {
  const ctx = await browser.newContext({ viewport: { width: m.width, height: m.height }, colorScheme: m.scheme });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.log(`[${m.name}] pageerror ${e.message}`));
  page.on("console", (msg) => msg.type() === "error" && console.log(`[${m.name}] console ${msg.text()}`));
  for (const [name, hash] of Object.entries(pages)) {
    await page.goto(base + hash);
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(1500);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    console.log(`${m.name} ${name}: overflow ${overflow}px`);
    await page.screenshot({ path: `${out}/${name}-${m.name}.png`, fullPage: true });
  }
  await ctx.close();
}
await browser.close();
