import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";

const [, , targetUrl, outputPath, rawWidth, rawHeight] = process.argv;

if (!targetUrl || !outputPath) {
  console.error("Usage: node capture-page.mjs <url> <output-path> [width] [height]");
  process.exit(2);
}

const width = Number.parseInt(rawWidth || "1440", 10);
const height = Number.parseInt(rawHeight || "1200", 10);

function chromiumCacheCandidates() {
  const root = join(process.env.HOME || "", ".cache", "ms-playwright");
  if (!root || !existsSync(root)) return [];
  return readdirSync(root)
    .filter((name) => name.startsWith("chromium-"))
    .sort()
    .reverse()
    .map((name) => join(root, name, "chrome-linux64", "chrome"));
}

function resolveExecutablePath() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
  return chromiumCacheCandidates().find((path) => existsSync(path));
}

let browser;
try {
  const executablePath = resolveExecutablePath();
  browser = await chromium.launch({
    headless: true,
    ...(executablePath ? { executablePath } : {}),
  });
  const page = await browser.newPage({
    viewport: { width, height },
    deviceScaleFactor: 1,
  });

  await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await page.waitForLoadState("networkidle", { timeout: 60_000 }).catch(() => {});
  await page
    .waitForSelector(".bench-table tbody tr, .empty, .error-bar, .admin-grid", { timeout: 60_000 })
    .catch(() => {});
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: outputPath, fullPage: true });

  const pageHeight = await page.evaluate(() =>
    Math.max(
      document.body.scrollHeight,
      document.documentElement.scrollHeight,
      document.body.offsetHeight,
      document.documentElement.offsetHeight
    )
  );
  console.log(JSON.stringify({ outputPath, pageHeight }));
} catch (error) {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
} finally {
  if (browser) {
    await browser.close();
  }
}
