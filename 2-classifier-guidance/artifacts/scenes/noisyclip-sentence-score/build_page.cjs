// Builds noisyclip-ladder.html: every rung's markdown page in one self-contained local file.
// Maths is rendered here with KaTeX, its fonts and every picture are embedded, and nothing loads from the network.
// Run from this folder: node build_page.cjs
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const NM = '/home/molef/.nvm/versions/node/v22.17.0/lib/node_modules/@marp-team/marp-cli/node_modules';
const MarkdownIt = require(path.join(NM, 'markdown-it'));
const katex = require(path.join(NM, 'katex'));
const L = __dirname;

const PAGES = [
  { id: 'start', file: 'read-through-a-small-network.md', nav: 'Start here', sub: 'the method on a small network' },
  { id: 'network', file: '0.5-network-view/README.md', nav: 'Network view', sub: 'what NoisyCLIP adds to CLIP' },
  { id: 'pictures', file: '1-illustrations/README.md', nav: 'Illustrations', sub: 'the idea in six pictures' },
  { id: 'miniature', file: '2-miniature/README.md', nav: 'Miniature', sub: 'work it by hand on 2 pixels' },
  { id: 'code', file: '3-code/walk-noisyclip.md', nav: 'Code walk', sub: 'the released code, block by block' },
  { id: 'rebuild', file: 'rebuild-from-fundamentals.md', nav: 'Rebuild', sub: 'each equation from fundamentals' },
  { id: 'notation', file: 'NOTATION.md', nav: 'Notation', sub: 'every symbol' },
  { id: 'about', file: 'README.md', nav: 'About', sub: 'the rungs and their files' },
];
const byFile = new Map(PAGES.map((p) => [path.join(L, p.file), p.id]));

const slug = (h) => h.replace(/\$([^$]*)\$/g, '$1').replace(/[`*_]|\[|\]\([^)]*\)/g, '').trim().toLowerCase()
  .replace(/[^\p{L}\p{N}_\- ]/gu, '').replace(/ /g, '-');

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function protectMaths(src) {
  const store = [];
  const re = /(```[\s\S]*?```)|(`[^`\n]+`)|\$\$([\s\S]+?)\$\$|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g;
  const out = src.replace(re, (m, fence, code, block, inline) => {
    if (fence || code) return m;
    const display = block !== undefined;
    store.push(katex.renderToString((display ? block : inline).trim(), { displayMode: display, throwOnError: true, strict: false, output: 'html' }));
    return `KATEXSLOT${store.length - 1}X`;
  });
  return { out, store };
}

function target(url, page) {
  if (/^(https?:|mailto:)/.test(url)) return { kind: 'web', url };
  if (url.startsWith('#')) return { kind: 'page', url: `#${page.id}--${url.slice(1)}` };
  const [p, a] = url.split('#');
  const abs = path.resolve(L, path.dirname(page.file), decodeURIComponent(p));
  if (byFile.has(abs)) return { kind: 'page', url: `#${byFile.get(abs)}${a ? `--${a}` : ''}` };
  const rel = path.relative(L, abs).split(path.sep).map(encodeURIComponent).join('/');
  return { kind: 'file', url: rel + (a ? `#${a}` : ''), abs };
}

const md = new MarkdownIt({ html: true, linkify: false, typographer: false });
const headingOpen = md.renderer.rules.heading_open || ((t, i, o, e, s) => s.renderToken(t, i, o));
md.renderer.rules.heading_open = (tokens, i, opts, env, self) => {
  const raw = env.restore(tokens[i + 1].content);
  tokens[i].attrSet('id', `${env.page.id}--${slug(raw)}`);
  return headingOpen(tokens, i, opts, env, self);
};

const wanted = new Set();
const rendered = PAGES.map((page) => {
  const raw = fs.readFileSync(path.join(L, page.file), 'utf8');
  const { out, store } = protectMaths(raw);
  const originals = [];
  const re = /(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g;
  raw.replace(/(```[\s\S]*?```)|(`[^`\n]+`)|\$\$([\s\S]+?)\$\$|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g, (m, f, c, b, i) => {
    if (!f && !c) originals.push(b !== undefined ? b : i);
    return m;
  });
  void re;
  const env = { page, restore: (s) => s.replace(/KATEXSLOT(\d+)X/g, (m, n) => `$${originals[+n]}$`) };
  let html = md.render(out, env);
  html = html.replace(/KATEXSLOT(\d+)X/g, (m, n) => store[+n]);
  html = html.replace(/<(a|img)\b([^>]*)>/g, (m, tag, attrs) => {
    const key = tag === 'a' ? 'href' : 'src';
    const found = attrs.match(new RegExp(`${key}="([^"]+)"`));
    if (!found || found[1].startsWith('data:')) return m;
    const t = target(found[1].replace(/&amp;/g, '&'), page);
    let next = attrs.replace(found[0], `${key}="${t.url}"`);
    if (tag === 'img') {
      if (t.kind === 'file') { wanted.add(t.abs); next = attrs.replace(found[0], `data-pic="${t.abs}"`); }
      next = next.replace(/\swidth="\d+"/, '');
      return `<img${next}>`;
    }
    if (t.kind === 'web') next += ' target="_blank" rel="noopener"';
    return `<a${next}>`;
  });
  return { page, html };
});

// Pictures are embedded once each as WebP, so a figure used on two pages does not double the file.
const py = `
import base64, io, json, sys
from PIL import Image
out = {}
for p in json.load(sys.stdin):
    im = Image.open(p).convert('RGB')
    im.thumbnail((1600, 1600))
    b = io.BytesIO()
    im.save(b, 'WEBP', quality=88, method=5)
    out[p] = 'data:image/webp;base64,' + base64.b64encode(b.getvalue()).decode()
json.dump(out, sys.stdout)
`;
const pics = JSON.parse(execFileSync('python3', ['-c', py], { input: JSON.stringify([...wanted]), maxBuffer: 1 << 29 }).toString());
const names = [...wanted];
const index = new Map(names.map((n, i) => [n, i]));
const body = rendered.map(({ page, html }) => {
  const h = html.replace(/data-pic="([^"]+)"/g, (m, p) => `data-p="${index.get(p)}"`);
  return `<section class="page" id="${page.id}">${h}</section>`;
}).join('\n');

let css = fs.readFileSync(path.join(NM, 'katex/dist/katex.min.css'), 'utf8');
css = css.replace(/src:([^;}]+)/g, (m, list) => {
  const w = list.match(/url\(fonts\/([^)]+\.woff2)\)/);
  if (!w) return m;
  const data = fs.readFileSync(path.join(NM, 'katex/dist/fonts', w[1])).toString('base64');
  return `src:url(data:font/woff2;base64,${data}) format("woff2")`;
});

const nav = PAGES.map((p) => `<a href="#${p.id}" data-page="${p.id}"><b>${esc(p.nav)}</b><span>${esc(p.sub)}</span></a>`).join('');
const page = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NoisyCLIP Zoom Ladder</title>
<style>${css}</style>
<style>
:root{color-scheme:light;--ink:#2b3440;--soft:#5f6b78;--line:#dfe3e8;--pale:#f6f7f9;--text:#b9a7e8;--image:#a8dcb5;--match:#bfe3f5;--push:#c86a0a;--fixed:#6b7a88}
*{box-sizing:border-box}html{scroll-padding-top:16px}
body{margin:0;background:#fff;color:var(--ink);font:17px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.shell{display:grid;grid-template-columns:262px minmax(0,1fr);min-height:100vh}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;border-right:1px solid var(--line);padding:22px 14px;background:var(--pale)}
nav h1{font-size:18px;line-height:1.25;margin:2px 10px 4px}nav p{font-size:13px;line-height:1.4;color:var(--soft);margin:0 10px 16px}
nav a{display:block;padding:8px 10px;border-radius:8px;color:var(--ink);text-decoration:none;border-left:3px solid transparent}
nav a b{display:block;font-size:15px;font-weight:600}nav a span{display:block;font-size:12.5px;line-height:1.3;color:var(--soft)}
nav a:hover{background:#eceef1}nav a.on{background:#fff;border-left-color:var(--fixed);box-shadow:inset 0 0 0 1px var(--line)}
.key{margin:18px 10px 0;font-size:12.5px;line-height:1.9;color:var(--soft)}.key i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:-1px}
main{min-width:0;padding:34px clamp(16px,4vw,56px) 90px}.page{max-width:1020px;margin:0 auto}
h1{font-size:clamp(27px,3.4vw,36px);line-height:1.2;margin:.1em 0 .6em}h2{font-size:25px;line-height:1.3;margin:1.9em 0 .55em;padding-top:.5em;border-top:1px solid var(--line)}h3{font-size:19px;margin:1.5em 0 .4em}
a{color:#2c6e91}p{margin:.75em 0}strong{font-weight:650}
p:has(> a > img){display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin:1.1em 0}p:has(> a > img) > a{flex:1 1 340px;min-width:0}
img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:zoom-in}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:.35em 0}.katex{font-size:1.06em}
table{display:block;overflow-x:auto;border-collapse:collapse;margin:1em 0;font-size:15px;max-width:100%}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--pale)}
code{font:.87em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--pale);padding:.1em .3em;border-radius:4px;overflow-wrap:anywhere}
pre{overflow-x:auto;background:#f7f8fa;border:1px solid var(--line);border-radius:8px;padding:12px 14px;line-height:1.45}pre code{background:none;padding:0;overflow-wrap:normal}
details{margin:1.5em 0;border:1px solid var(--line);border-radius:8px;padding:8px 14px;background:var(--pale)}summary{cursor:pointer;font-weight:600}
#box{position:fixed;inset:0;display:none;align-items:center;justify-content:center;padding:18px;background:rgba(32,38,46,.82);z-index:10;cursor:zoom-out}#box.on{display:flex}#box img{max-width:100%;max-height:100%;width:auto;border:0;cursor:zoom-out}
@media (max-width:860px){.shell{display:block}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:14px 10px}nav p,.key{display:none}nav h1{margin-bottom:8px}.links{display:flex;gap:6px;overflow-x:auto}.links a{flex:none;padding:6px 10px}.links a span{display:none}main{padding:22px 16px 70px}}
</style></head><body><div class="shell">
<nav><h1>NoisyCLIP as a sentence-score push</h1><p>A zoom ladder. No result on this project's pictures exists yet.</p><div class="links">${nav}</div>
<div class="key"><div><i style="background:var(--text)"></i>text</div><div><i style="background:var(--image)"></i>image</div><div><i style="background:var(--match)"></i>matching pair</div><div><i style="background:var(--push)"></i>the push</div><div><i style="background:var(--fixed)"></i>frozen or fixed</div></div></nav>
<main>${body}</main></div><div id="box" role="dialog" aria-label="Enlarged figure"><img alt=""></div>
<script id="pics" type="application/json">${JSON.stringify(names.map((n) => pics[n]))}</script>
<script>
const P=JSON.parse(document.getElementById('pics').textContent);
document.querySelectorAll('img[data-p]').forEach(i=>{i.src=P[+i.dataset.p]});
const pages=[...document.querySelectorAll('.page')],links=[...document.querySelectorAll('nav a')],box=document.getElementById('box'),big=box.querySelector('img');
function go(){const h=decodeURIComponent(location.hash.slice(1));const id=(h.split('--')[0])||'start';const p=pages.find(x=>x.id===id)||pages[0];
pages.forEach(x=>x.hidden=x!==p);links.forEach(a=>a.classList.toggle('on',a.dataset.page===p.id));
const t=h.includes('--')?document.getElementById(h):null;if(t)t.scrollIntoView();else window.scrollTo(0,0)}
addEventListener('hashchange',go);go();
document.addEventListener('click',e=>{const a=e.target.closest('a');if(a&&a.querySelector(':scope > img')){e.preventDefault();big.src=a.querySelector('img').src;big.alt=a.querySelector('img').alt;box.classList.add('on')}});
box.addEventListener('click',()=>box.classList.remove('on'));addEventListener('keydown',e=>{if(e.key==='Escape')box.classList.remove('on')});
</script></body></html>`;
fs.writeFileSync(path.join(L, 'noisyclip-ladder.html'), page);
console.log(`noisyclip-ladder.html: ${(page.length / 1048576).toFixed(1)} MB, ${PAGES.length} pages, ${names.length} pictures embedded once each`);
