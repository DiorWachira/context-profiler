import { expect, test, type Page } from "@playwright/test";

// Assumes the JobBridge demo: `ctxprof demo` (prune 1 = default, prune 2 = --keep report_listing).

async function noOverflow(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
}

test("overview -> trace -> keyboard zoom -> marker readout", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./#/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("tokens in.");
  await expect(page.getByText("Verified saving / 100k req")).toBeVisible();

  await page.getByRole("link", { name: "Open a flame graph" }).click();
  await expect(page).toHaveURL(/#\/trace\//);
  const ruler = page.locator("button.ruler");
  const before = await ruler.getAttribute("aria-label");
  const stack = page.getByRole("group", { name: /Flame stack/ });
  await stack.focus();
  await page.keyboard.press("w");
  await page.keyboard.press("w");
  await expect(ruler).not.toHaveAttribute("aria-label", before!);
  await page.keyboard.press("0");
  await expect(ruler).toHaveAttribute("aria-label", before!);

  await ruler.click({ position: { x: 80, y: 18 } });
  await ruler.click({ position: { x: 400, y: 18 } });
  const delta = page.locator(".readouts").filter({ hasText: "Δ tokens" });
  await expect(delta).toContainText("Σ influence inside");
  const tokens = await delta.locator(".v").nth(2).innerText();
  expect(Number(tokens.replace(/,/g, ""))).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test("segments table sorts by influence per 1k tokens", async ({ page }) => {
  await page.goto("./#/segments");
  const th = page.locator("th", { hasText: "Infl / 1k tok" });
  await expect(page.locator("th[aria-sort=descending]")).toContainText("Tokens paid");
  await th.getByRole("button").click();
  await expect(th).toHaveAttribute("aria-sort", "descending");
  const vals = await page.locator("tbody tr td:nth-child(9)").allInnerTexts();
  const nums = vals.map(Number);
  expect(nums).toEqual([...nums].sort((a, b) => b - a));
});

test("pruning: toggling a hunk recomputes the verdict from held-out evidence", async ({ page }) => {
  await page.goto("./#/pruning/2");
  const stamp = page.locator(".stamp");
  await expect(stamp).toHaveText(/VERIFIED · RECOMMEND/);
  await expect(page.locator(".vgrid i.bad")).toHaveCount(0);

  const cut = page.getByRole("checkbox", { name: "Cut tool/report_listing" });
  await expect(cut).not.toBeChecked();
  await cut.check(); // keyboard-reachable native checkbox
  await expect(stamp).toHaveText(/ESTIMATE FAILS/);
  await expect(page.locator(".vgrid i.bad")).toHaveCount(6);
  await expect(page.locator(".reg li").first()).toContainText("tool/report_listing");

  await page.goto("./#/pruning/1");
  await expect(page.locator(".stamp")).toHaveText(/BELOW THRESHOLD/);
  const cut1 = page.getByRole("checkbox", { name: "Cut tool/report_listing" });
  await cut1.focus();
  await page.keyboard.press("Space");
  await expect(page.locator(".stamp")).toHaveText(/ESTIMATE PASSES/);
  await expect(page.locator(".reg li.ok")).toHaveCount(6);
});

test("pruning ledger converts to KSh at a configurable rate", async ({ page }) => {
  await page.goto("./#/pruning/2");
  const ksh = page.locator(".ledger .sm").filter({ hasText: "KSh" }).first();
  const rate = page.getByRole("spinbutton", { name: "KSh per US dollar" });
  await rate.fill("100");
  await expect(ksh).toHaveText(/≈ KSh 66,900/); // $669.03 * 100, rounded to KSh 100
});

test("375 px: no horizontal overflow on any screen", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  for (const hash of ["#/", "#/segments", "#/pruning/2", "#/trace/t0213"]) {
    await page.goto("./" + hash);
    await page.locator("main").first().waitFor();
    await page.waitForLoadState("networkidle");
    await noOverflow(page);
  }
});

test("prefers-reduced-motion disables the sweep", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await page.goto(test.info().project.use.baseURL + "#/trace/t0001");
  const sweep = page.locator(".sweep");
  await expect(sweep).toHaveCount(1);
  expect(await sweep.evaluate((el) => getComputedStyle(el).animationName)).toBe("none");
  await ctx.close();
});

test("theme toggle switches and persists", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("./#/");
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe("rgb(220, 222, 225)");
});

test("trace list filters and search", async ({ page }) => {
  await page.goto("./#/");
  const count = page.locator("#tr-h ~ span");
  await expect(count).toHaveText("260 of 260");
  await page.getByRole("group", { name: "Filter by status" }).getByRole("button", { name: "held out" }).click();
  await expect(count).toHaveText("60 of 260");
  await page.getByRole("group", { name: "Filter by status" }).getByRole("button", { name: "⚑ injection" }).click();
  await expect(count).toHaveText("8 of 260");
  await page.getByRole("group", { name: "Filter by intent" }).getByRole("button", { name: "status" }).click();
  await expect(page.getByText("No trace matches these filters.")).toBeVisible();
  await page.getByRole("group", { name: "Filter by status" }).getByRole("button", { name: "all" }).click();
  await page.getByRole("group", { name: "Filter by intent" }).getByRole("button", { name: "all" }).click();
  await page.getByRole("searchbox").fill("t0014");
  await expect(count).toHaveText("1 of 260");
  await page.getByRole("link", { name: "t0014", exact: true }).click();
  await expect(page).toHaveURL(/#\/trace\/t0014$/);
});
