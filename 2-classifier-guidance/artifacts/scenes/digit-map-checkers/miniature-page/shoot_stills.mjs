// Stills of the miniature page, light and dark, at rest and frozen mid-animation.
// Uses the page's window.__seek(phase, fraction) so each animated still is the same every run.
// Run: node shoot_stills.mjs   (writes stills/<name>-<theme>.png and prints what each shows)
import { createRequire } from "node:module";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const req = createRequire(path.join(os.homedir(), ".cache/learning-render", "package.json"));
const { chromium } = req("playwright-core");
const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAGE = "file://" + path.join(HERE, "digit-map-checkers-miniature.html");
const OUT = path.join(HERE, "stills");
fs.mkdirSync(OUT, { recursive: true });

// [file name, frame id, setup run in the page, what it shows]
const SHOTS = [
  ["01-forward-mid-convolution", "logit", "__seek(3, 0.55)", "forward pass, packets travelling from the maps into the convolution"],
  ["02-forward-at-rest", "logit", "", "forward pass complete, logit 0.772 on the apart canvas"],
  ["03-training-into-the-kernel", "training", "__seek(12, 0.5)", "training backward pass, orange packets running back into the kernel's weights"],
  ["04-training-stops-at-maps", "training", "__seek(14, 0)", "training backward pass ended: kernel gradients on the edges, stop marker at the maps"],
  ["05-push-through-the-unet", "push", "__seek(12, 0.5)", "push backward pass, packets leaving the maps into the frozen U-Net"],
  ["06-push-into-the-pixels", "push", "__seek(13, 0.6)", "push backward pass reaching the four pixels"],
  ["07-push-broken-run", "push", "document.getElementById('broken').click()", "broken run: maps built inside no_grad, g = 0 on every pixel"],
  ["08-noise-sweep-at-712", "noise", "__seek(0, 0)", "noise sweep at its first timestep, t = 712"],
  ["09-push-length-at-1", "push-length", "__seek(20, 0)", "push-length sweep at b = 1.00"],
  ["10-identity-training", "training", "document.getElementById('br-identity').click()", "identity branch after one step: both logits fall, gap 0.4205 to 0.4192"],
];

const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome", args: ["--no-sandbox"] });
const report = [];
for (const theme of ["light", "dark"]) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 860 }, colorScheme: theme, reducedMotion: "no-preference" });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(String(e.message || e)));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());   // headless from WSL stalls on font hosts
  for (const [name, frame, setup, shows] of SHOTS) {
    await page.goto(PAGE + "?shot=" + name + "-" + theme + "#" + frame, { waitUntil: "load" });
    await page.waitForTimeout(150);
    if (setup) await page.evaluate(setup);
    await page.waitForTimeout(120);
    const file = path.join(OUT, `${name}-${theme}.png`);
    await page.screenshot({ path: file });
    const check = await page.evaluate(() => window.__check);
    report.push({ file: path.relative(HERE, file), shows, check_ok: check && check.ok });
  }
  if (errors.length) report.push({ theme, errors });
  await ctx.close();
}
await browser.close();
console.log(JSON.stringify(report, null, 1));
