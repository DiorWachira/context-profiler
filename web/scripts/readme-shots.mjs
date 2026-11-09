// README screenshots (viewport, not full page): node scripts/readme-shots.mjs [baseUrl]
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:4173/";
const out = "../docs/img";
mkdirSync(out, { recursive: true });
const shots = [
  { file: "overview-dark.png", hash: "#/", w: 1440, h: 900, scheme: "dark" },
  { file: "trace-light.png", hash: "#/trace/t0014", w: 1440, h: 900, scheme: "light" },
  { file: "pruning-dark.png", hash: "#/pruning/1", w: 1440, h: 900, scheme: "dark" },
  { file: "pruning-phone.png", hash: "#/pruning/2", w: 390, h: 844, scheme: "light" },
];

const browser = await chromium.launch();
for (const s of shots) {
  const ctx = await browser.newContext({ viewport: { width: s.w, height: s.h }, colorScheme: s.scheme, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(base + s.hash);
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(1600); // let the heat sweep finish
  await page.screenshot({ path: `${out}/${s.file}` });
  await ctx.close();
}
await browser.close();
console.log(`wrote ${shots.length} screenshots to ${out}`);
