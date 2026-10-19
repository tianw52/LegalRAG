// End-to-end UI smoke test: drives the installed Chrome through
// login → onboarding → home → one live exploration → one precomputed
// example (demo) with the Ask panel, saving a screenshot per step and
// reporting page errors.
//
// Quick mode (default) stops the live run once retrieval-backed tabs render
// (Overview progress, Sources reader, Timeline).  FULL=1 also waits for the
// LLM views (What-if, Relations) and opens "under the hood" — slow with a
// local LLM.  The demo step needs data/demos/ (scripts/build_demos.py); its
// Ask-panel question needs the LLM (ASK=0 skips it).
//
// Prereqs: API on :8000, `npm run dev` on :5173.
// Usage:   npm run smoke
//          FULL=1 WAIT_MIN=20 QUESTION="..." PROVINCE="Ontario" OUT=../logs/ui-smoke npm run smoke
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const BASE_URL = process.env.BASE_URL || "http://localhost:5173";
const CHROME =
  process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const OUT = path.resolve(process.env.OUT || "../logs/ui-smoke");
const FULL = process.env.FULL === "1";
const ASK = process.env.ASK !== "0";
const PROVINCE = process.env.PROVINCE || "British Columbia"; // tile name, or a non-province choice label
const WAIT_MS = Number(process.env.WAIT_MIN || 20) * 60_000;
const QUESTION =
  process.env.QUESTION ||
  "I quit my part-time job for a better-paying summer job that ended. Can I still get Employment Insurance, or will they say I left without just cause?";

fs.mkdirSync(OUT, { recursive: true });
const problems = [];

async function shot(page, name) {
  const file = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: file, fullPage: true });
  console.log(`  📸 ${file}`);
}

// Click the first element matching `selector` whose text contains `text`.
async function clickText(page, selector, text) {
  const ok = await page.evaluate(
    (sel, t) => {
      const el = [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(t));
      if (el) el.click();
      return !!el;
    },
    selector,
    text
  );
  if (!ok) throw new Error(`No ${selector} containing "${text}"`);
}

// Case-insensitive: display headings are uppercased by CSS (innerText is rendered text).
async function waitText(page, text, timeout = 60_000) {
  await page.waitForFunction(
    (t) => document.body.innerText.toLowerCase().includes(t.toLowerCase()),
    { timeout },
    text
  );
}

async function tab(page, label, name) {
  await clickText(page, "nav.tabs .tab", label);
  await new Promise((r) => setTimeout(r, 400));
  await shot(page, name);
}

// Wait for the API (through the Vite proxy) so a restarting backend doesn't fail step 1.
async function waitForApi(timeoutMs = 180_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${BASE_URL}/api/health`);
      if (res.ok) return console.log(`API ready: ${JSON.stringify(await res.json())}`);
    } catch {
      /* dev server or API not up yet */
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error(`API not reachable at ${BASE_URL}/api/health`);
}

await waitForApi();
// protocolTimeout must exceed the longest single wait (LLM views can take minutes locally)
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  protocolTimeout: WAIT_MS + 60_000,
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000 });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && problems.push(`console: ${m.text()}`));

  console.log("1. login");
  await page.goto(BASE_URL, { waitUntil: "networkidle0" });
  await page.evaluate(() => localStorage.clear());
  await page.reload({ waitUntil: "networkidle0" });
  await shot(page, "01-login");
  await page.type(".login input", `Smoke Test ${Date.now() % 10000}`);
  await page.click(".login .go-btn");

  console.log("2. onboarding");
  await waitText(page, "Where are you?");
  const tile = await page.$(`.tiles .tile[aria-label="${PROVINCE}"]`);
  if (tile) await tile.click();
  else await clickText(page, ".onboard-step .seg", PROVINCE);
  await waitText(page, PROVINCE.replace(/^Explore /, ""));
  await shot(page, "02-onboarding-location");
  await clickText(page, "button", "Next");
  await clickText(page, ".opt", "New to law");
  await shot(page, "03-onboarding-familiarity");
  await clickText(page, "button", "Next");
  await clickText(page, ".opt", "specific situation");
  await clickText(page, "button", "Next");
  await clickText(page, ".chip", "Employment");
  await shot(page, "04-onboarding-topics");
  await clickText(page, "button", "Start exploring");

  console.log("3. home");
  await page.waitForSelector(".hero textarea");
  await page.waitForSelector(".band .rows .rowlink"); // benchmark examples
  await shot(page, "05-home");

  console.log("4. explore live (retrieval-backed tabs)");
  await page.type(".hero textarea", QUESTION);
  await page.click(".hero .go-btn");
  await page.waitForSelector(".pp", { timeout: 60_000 });
  await shot(page, "06-overview-building");
  await page.waitForSelector(".top-sources", { timeout: 300_000 });
  await tab(page, "Sources", "07-sources");
  await page.waitForSelector(".reader-text mark", { timeout: 60_000 });
  await page.click(".doc-list li:nth-child(2) .doc-row");
  await page.waitForSelector(".reader-text mark", { timeout: 60_000 });
  await shot(page, "08-sources-reader");
  await tab(page, "Timeline", "09-timeline");

  if (FULL) {
    console.log(`5. LLM views (waiting up to ${WAIT_MS / 60000} min)…`);
    await page.waitForSelector(".done-line", { timeout: WAIT_MS }).catch(async () => {
      await tab(page, "Overview", "10-overview-waiting");
      throw new Error("run did not finish");
    });
    await tab(page, "Overview", "10-overview-done");
    await tab(page, "What-if", "11-whatif");
    await tab(page, "Relations", "12-relations");
    await tab(page, "Overview", "10-overview-done");
    await clickText(page, ".asked-meta .link", "Under the hood");
    await page.waitForSelector(".drawer-body.trace .llm-call", { timeout: 30_000 });
    await shot(page, "13-trace");
    await page.keyboard.press("Escape");
  }

  console.log("6. precomputed example (demo)");
  await page.click(".topbar .new-q"); // New question
  await page.waitForSelector(".band .rows .rowlink");
  await page.click(".band .rows .rowlink");
  await page.waitForSelector(".brief", { timeout: 60_000 });
  const demo = await page.$(".done-line");
  console.log(demo ? "  demo replayed (no progress panel)" : "  no demo bundle — ran live");
  await shot(page, "14-demo-overview");
  if (demo) {
    const answer = await page.$(".narrow .ans");
    if (answer) {
      await clickText(page, ".narrow .ans", "Not true");
      await shot(page, "15-demo-narrow-answered");
    }
    await tab(page, "What-if", "16-demo-whatif");
    await tab(page, "Relations", "17-demo-relations");
    await tab(page, "Sources", "18-demo-sources");
    await page.click(".ask-btn");
    await page.waitForSelector(".chat");
    await shot(page, "19-demo-ask-panel");
    if (ASK) {
      await clickText(page, ".chat .sugg", "Summarise");
      await page.waitForFunction(() => document.querySelector(".msg-ai") && !document.querySelector(".msg-ai .pending"), { timeout: WAIT_MS });
      await shot(page, "20-demo-ask-answer");
    }
  }
} catch (e) {
  problems.push(`smoke step failed: ${e.message}`);
} finally {
  await browser.close();
}

if (problems.length) {
  console.log("\nProblems:\n - " + problems.join("\n - "));
  process.exit(1);
}
console.log("\nSmoke test passed.");
