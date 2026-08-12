// Drive the real page in a real browser, one model at a time.
//
//   npm install && npm run vendor
//   python -m sodachat.export_onnx
//   python -m sodachat.web --port 8733 &
//   node tools/check_browser.mjs                    # every exported model
//   node tools/check_browser.mjs chat expert-route  # or a few
//
// The other checks (`check:tokenizer`, `check:model`) run the browser's *code*
// under Node, which is most of what matters and none of what the browser adds:
// module resolution, WASM instantiation, cross-origin isolation, the DOM. Both
// bugs this caught on its first run lived in exactly that gap — the vendored
// runtime naming the wrong WASM file, and a relative `wasmPaths` that resolves
// as a module specifier and cannot be found.
//
// Optional, and skips cleanly rather than failing when the pieces are missing:
// it needs `npm install puppeteer-core` and a Chrome, found via $CHROME_PATH or
// the usual install locations.

import { existsSync } from "node:fs";

const URL_BASE = process.env.SODACHAT_URL ?? "http://127.0.0.1:8733/";
const CHROMES = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium-browser",
  "/usr/bin/chromium",
].filter(Boolean);

const skip = (why) => {
  console.log(`skipping browser check — ${why}`);
  process.exit(0);
};

let puppeteer;
try {
  puppeteer = (await import("puppeteer-core")).default;
} catch {
  skip("puppeteer-core is not installed (npm install puppeteer-core)");
}
const executablePath = CHROMES.find((p) => existsSync(p));
if (!executablePath) skip("no Chrome found (set $CHROME_PATH)");

const manifest = await fetch(new URL("models/manifest.json", URL_BASE))
  .then((r) => r.json())
  .catch(() => skip(`nothing serving ${URL_BASE} (python -m sodachat.web --port 8733)`));

const wanted = process.argv.slice(2);
const models = wanted.length ? wanted : Object.keys(manifest.models);

const browser = await puppeteer.launch({
  executablePath,
  headless: true,
  // SwiftShader stands in for a GPU on a headless box, so the WebGPU path is
  // exercised rather than silently skipped.
  args: ["--enable-unsafe-swiftshader", "--no-sandbox"],
});

let failed = 0;
for (const model of models) {
  const problems = [];
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && problems.push(m.text()));
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => problems.push(`${r.url()} failed`));

  try {
    await page.goto(URL_BASE, { waitUntil: "domcontentloaded", timeout: 30_000 });
    await page.waitForFunction("document.getElementById('model').options.length > 0", {
      timeout: 20_000,
    });
    await page.select("#model", model);
    await page.click("#load");
    await page.waitForFunction("!document.getElementById('workspace').hidden", {
      timeout: 300_000,
    });
    const tabs = await page.$$eval("#tabs button", (b) => b.map((x) => x.textContent));
    const result = await exercise(page, tabs);
    const backend = (await page.$eval("#backend-hint", (e) => e.textContent)).trim();
    console.log(`${model.padEnd(15)} ${backend.padEnd(22)} ${tabs.join("/")} → ${result}`);
  } catch (error) {
    problems.push(error.message);
  }
  await page.close();
  if (problems.length) {
    failed++;
    console.error(`FAIL ${model}\n  ${problems.join("\n  ")}`);
  }
}

await browser.close();
console.log(failed ? `${failed}/${models.length} models failed` : `${models.length} models OK`);
process.exit(failed ? 1 : 0);

/** Use whichever panel this model's `kind` produced, and report what came back. */
async function exercise(page, tabs) {
  await page.click("#tabs button");
  if (tabs.includes("Chat")) {
    await page.type("#message", "hey there, what did you do today?");
    await page.click("#send");
    await page.waitForFunction(
      "document.querySelector('.turn.bot') && !document.getElementById('send').disabled",
      { timeout: 300_000 },
    );
    return JSON.stringify(await page.$eval(".turn.bot", (e) => e.childNodes[1].textContent));
  }
  if (tabs.includes("Classify")) {
    await page.type("#classify-input", "what is 12 * 7?");
    await page.click("#classify-run");
    await page.waitForFunction(
      "document.getElementById('classify-out').textContent.includes('%')",
      { timeout: 120_000 },
    );
    return page.$eval("#classify-out", (e) => e.textContent);
  }
  if (tabs.includes("Generate")) {
    await page.type("#generate-input", "def add(a, b):");
    await page.click("#generate-run");
    await page.waitForFunction(
      "document.getElementById('generate-status').textContent.includes('tokens')",
      { timeout: 300_000 },
    );
    return page.$eval("#generate-status", (e) => e.textContent);
  }
  if (tabs.includes("Play snake")) {
    await page.click("#play-step");
    await page.waitForFunction("document.getElementById('board').textContent.includes('@')", {
      timeout: 120_000,
    });
    const rows = await page.$eval("#board", (e) => e.textContent.split("\n").length);
    return `${rows}-row board, ${await page.$eval("#play-status", (e) => e.textContent)}`;
  }
  throw new Error(`no panel for tabs ${tabs}`);
}
