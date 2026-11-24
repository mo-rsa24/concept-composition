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
  ["00a-real-input", "real-system", "__real('apart_1_3', 480, 0)", "opening frame, red box on the noisy canvas: apart 1|3 at t = 480, clean and noisy side by side"],
  ["00b-real-maps", "real-system", "__real('apart_1_3', 480, 2)", "opening frame, red box on the two real maps"],
  ["00c-real-conv1", "real-system", "__real('apart_1_3', 480, 3)", "opening frame, conv 1: 8 channels before and after SiLU"],
  ["00d-real-film", "real-system", "__real('apart_1_3', 480, 7)", "opening frame, the timestep's scale and shift, with the 128 gammas and betas"],
  ["00e-real-pool", "real-system", "__real('apart_1_3', 480, 9)", "opening frame, the 256 pooled numbers as two strips and the 16 time features"],
  ["00f-real-logit-wrong", "real-system", "__real('wrong_1_3', 616, 10)", "opening frame, the logit on wrong 3|2 at t = 616: -1.115, the checker says no"],
  ["00g-real-push", "real-system", "__real('apart_3_4', 539, 11)", "opening frame, the real push gradient on apart 3|4 at t = 539"],
  ["00h-real-unet-712", "real-system", "__real('wrong_3_4', 712, 1)", "opening frame, three noise guesses on wrong 2|3 at the noisiest level"],
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
  ["11-noise-real-scores", "noise", "", "noise frame at t = 480: the real count checker's 0.63 and 0.92 ends and its 0.791 average"],
  ["00i-real-readout-network", "real-system", "__real('apart_3_4', 539, 10)", "opening frame, the readout MLP as nodes on apart 3|4 at t = 539: 12 hidden units, their activations and output weights, and the sum that gives the logit 1.297"],
  ["13-push-score-card", "push", "document.querySelectorAll('#pbody .card')[1].scrollIntoView()", "push frame, the card that reads the push as classifier guidance: (1 - p) drops out, and subtracting along g adds s_t g to the score"],
  ["14-two-digits-before-push", "two-digits", "__trace('6-8', 'won', 10)", "step-by-step frame, digits 6 and 8, seed 1000, step 10: no push yet, the two lanes are one run"],
  ["15-two-digits-push-acting", "two-digits", "__trace('6-8', 'won', 20)", "step 20, push acting: pushed maps 9.1 px apart against 3.1 in the plain lane; logit 3.78 against -1.08"],
  ["16-two-digits-end", "two-digits", "__trace('5-6', 'won', 49)", "digits 5 and 6, seed 1006, step 49: plain ends as one blob, pushed as two clear digits"],
  ["17-two-digits-lost", "two-digits", "__trace('7-9', 'lost', 49)", "digits 7 and 9, seed 1000, step 49: a seed the push did not separate, both lanes end as one blob"],
  ["12-convolution-hover", "convolution", "(() => { const h = document.querySelector('.hit[data-edge=k0000]'); const r = h.getBoundingClientRect(); h.dispatchEvent(new PointerEvent('pointerenter', {clientX: r.left + r.width * 0.6, clientY: r.top + r.height * 0.6, bubbles: true})); })()", "convolution frame with the kernel lines bare and one weight's hover label open"],
];
// phone width: the opening frame and the network, where crowding would show first
const PHONE = [
  ["p1-real-conv1", "real-system", "__real('apart_1_3', 480, 3)"],
  ["p2-real-logit", "real-system", "__real('wrong_1_3', 616, 10)"],
  ["p3-network-forward", "logit", ""],
  ["p4-training-end", "training", "__seek(14, 0)"],
  ["p5-real-readout", "real-system", "__real('apart_3_4', 539, 10)"],
  ["p6-two-digits", "two-digits", "__trace('6-8', 'won', 20)"],
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
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, colorScheme: theme, hasTouch: true });   // no isMobile: the page has no viewport meta until the host wraps it
  const pp = await phone.newPage();
  await pp.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  for (const [name, frame, setup] of PHONE) {
    await pp.goto(PAGE + "?shot=" + name + "-" + theme + "#" + frame, { waitUntil: "load" });
    await pp.waitForTimeout(150);
    if (setup) await pp.evaluate(setup);
    await pp.waitForTimeout(120);
    const file = path.join(OUT, `${name}-${theme}.png`);
    await pp.screenshot({ path: file, fullPage: true });
    report.push({ file: path.relative(HERE, file), phone: true, scrollWidth: await pp.evaluate(() => document.documentElement.scrollWidth) });
  }
  await phone.close();
}
await browser.close();
console.log(JSON.stringify(report, null, 1));
