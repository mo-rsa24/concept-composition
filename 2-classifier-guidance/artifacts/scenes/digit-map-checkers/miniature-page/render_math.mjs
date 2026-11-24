// Render every equation in equations.json with KaTeX 0.16.11 (installed in ./node_modules, git-ignored).
// Writes JSON {name: html} to stdout; build_page.py inlines it so the page typesets nothing at load time.
// Run by build_page.py: node render_math.mjs equations.json
import fs from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const katex = require("katex");
if (katex.version !== "0.16.11") throw new Error("expected KaTeX 0.16.11, found " + katex.version);
const src = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = {};
for (const [name, v] of Object.entries(src)) {
  if (name.startsWith("_")) continue;
  const [mode, tex] = v;
  out[name] = katex.renderToString(tex, { displayMode: mode === "d", output: "htmlAndMathml", throwOnError: true, strict: "error" });
}
process.stdout.write(JSON.stringify(out));
