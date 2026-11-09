// Screenshots of the guided tour: node scripts/tour-shots.mjs [baseUrl] [outDir]
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:8000/";
const out = process.argv[3] ?? "shots/tour";
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
for (const m of [{ name: "desk", w: 1440, h: 900, scheme: "dark" }, { name: "phone", w: 390, h: 844, scheme: "light" }]) {
  const ctx = await browser.newContext({ viewport: { width: m.w, height: m.h }, colorScheme: m.scheme });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.log(`[${m.name}] pageerror ${e.message}`));
  await page.goto(base + "#/");
  await page.getByRole("button", { name: "Start the guide" }).click();
  for (let i = 1; i <= 12; i++) {
    await page.waitForTimeout(900);
    if ([1, 3, 5, 8, 10, 12].includes(i)) await page.screenshot({ path: `${out}/${m.name}-${String(i).padStart(2, "0")}.png` });
    const title = await page.locator(".callout h2").innerText();
    console.log(`${m.name} step ${i}: ${title} @ ${new URL(page.url()).hash}`);
    if (i < 12) await page.keyboard.press("ArrowRight");
  }
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: /Guide: open/ }).click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${out}/${m.name}-manual.png` });
  await ctx.close();
}
await browser.close();
