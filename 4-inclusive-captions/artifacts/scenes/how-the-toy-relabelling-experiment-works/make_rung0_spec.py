"""Rung 0's spec: the toy caption model's high-level design, every number read from
toy_relabelling_numbers.json and none typed. Grown from subproject 3's version 1
(3-understanding-compositionality/artifacts/scenes/how-the-caption-model-works/the-caption-model-high-level-design-explainer.json):
the same 23 boxes in the same grid cells, with the wording fixed to the numbers file, a colour role per box
from ladder.json's colour key, and picture cards placed from pictures.json.

    python3 make_rung0_spec.py [--pictures the-toy-caption-model-pictures/pictures.json]

Writes the-toy-caption-model-high-level-design-explainer.json beside this script.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.load(open(HERE / "toy_relabelling_numbers.json"))
STEM = "the-toy-caption-model-high-level-design-explainer"
P, TS, V, D, S, SM, C, SC = (N[k] for k in ("pictures", "training set", "vae", "denoiser", "schedule", "sampler", "corrector", "scorer"))
f = lambda v: f"{v:,}"
g = lambda v: f"{v:g}"

PIX = f"{P['height']} x {P['width']} x {P['channels']}"
LAT = " x ".join(map(str, V["latent shape"]))
PER = f(TS["pictures per caption before relabelling"])
IDS = TS["caption ids"]
EMPTY_PCT = g(TS["empty caption share of training draws"] * 100)
STEPS = S["reverse steps"]
GUIDE = g(SM["guidance strength"])
CS = C["settings (steps per level, step as share of noise variance)"]
ETA = g(CS["main"][1])
XP = P["positions (centre x, pixels)"]
CAPT = " x ".join(map(str, D["caption table"]))
k0 = TS["by relabelled count"]["0"]["training pictures"]
kmax = max(int(k) for k in TS["by relabelled count"])

nodes = [
    # lane 1: training pictures and captions
    dict(id="both", kind="files", role="inputs", eyebrow="pictures", label=f"{PER} both shapes",
         sub=f"caption id {IDS['a red circle and a blue square']}: \"a red circle and\na blue square\"; circle left and\nsquare right, or the mirror", col=0, row=0.3),
    dict(id="relabel", kind="op", role="inputs", label="Relabel k pictures",
         sub=f"copy k both-shapes pictures\nonce under id {IDS['a red circle']}, once under id {IDS['a blue square']}\n(k from 0 to {f(kmax)})", col=1, row=0.3, h=165),
    dict(id="circles", kind="files", role="inputs", eyebrow="pictures", label=f"{PER} red circles",
         sub=f"caption id {IDS['a red circle']}: \"a red circle\"\nat x = {g(XP['left'])}, {g(XP['centre'])} or {g(XP['right'])}", col=1, row=1.15),
    dict(id="squares", kind="files", role="inputs", eyebrow="pictures", label=f"{PER} blue squares",
         sub=f"caption id {IDS['a blue square']}: \"a blue square\"\nat x = {g(XP['left'])}, {g(XP['centre'])} or {g(XP['right'])}", col=1, row=2.0),
    dict(id="trainset", kind="store", role="inputs", eyebrow="training set", label=f"{f(k0)} + 2k pictures",
         sub=f"each {PIX}, with a caption id,\nplus pixel noise of sd {g(P['pixel noise sd added to every training copy'])}", col=2, row=1.15),
    # lane 2: the VAE
    dict(id="pic", kind="tensor", role="inputs", label="", sub=f"Picture: {PIX} = {f(P['numbers per picture'])} numbers", cells="8x4", col=0, row=3.35),
    dict(id="enc", kind="encoder", role="plain", label="VAE encoder", sub=f"convolutions, down to {LAT}", col=1, row=3.35),
    dict(id="latent", kind="tensor", role="hidden", label="", sub=f"Latent z0: {V['latent numbers']} numbers ({LAT})", cells="8x4", col=2, row=3.35),
    dict(id="dec", kind="decoder", role="plain", label="VAE decoder", sub=f"{LAT} back to {PIX}", col=3, row=3.35),
    dict(id="recon", kind="tensor", role="outputs", label="", sub=f"Picture again: {PIX}", cells="8x4", col=4, row=3.35),
    # lane 3: the denoiser
    dict(id="noise", kind="op", role="hidden", label="Add noise at level t",
         sub=f"z_t: t from 1 to {STEPS}, {S['kind']} schedule;\nt also goes in as {D['time features']} sines, cosines", col=2, row=4.85, h=135),
    dict(id="capid", kind="block", role="inputs", eyebrow="caption id", label=f"One of {', '.join(str(v) for v in IDS.values())}",
         sub=f"id {IDS['empty']} is the empty caption,\nused on {EMPTY_PCT}% of training draws", col=2, row=5.75),
    dict(id="lookup", kind="block", role="hidden", eyebrow="lookup table", label="Caption vector",
         sub=f"a learned {CAPT} table, one row per id;\nno text encoder reads any words", col=3, row=5.75),
    dict(id="mlp", kind="block", role="hidden", eyebrow="fully connected", label="Denoiser",
         sub=f"{D['input width']} numbers in, {D['output width']} out;\n{D['kind'].split(', ')[1]} of {f(D['hidden width'])};\n{f(D['parameters'])} parameters", col=3, row=4.85),
    dict(id="est", kind="tensor", role="outputs", label="", sub=f"Guess z0_hat: {D['output width']} numbers", cells="8x4", col=4, row=4.85),
    dict(id="loss", kind="loss", role="plain", label="Squared error",
         sub=f"z0_hat against the clean z0;\n{f(D['training steps'])} steps, batch {D['batch']}", col=5, row=4.85),
    # lane 4: sampling with the product
    dict(id="start", kind="tensor", role="inputs", label="", sub=f"Random start: {V['latent numbers']} numbers", cells="8x4", col=0, row=7.25),
    dict(id="calls", kind="op", role="outputs", label=f"{SM['denoiser calls per step for the product']} denoiser calls",
         sub=f"empty, id {IDS['a red circle']} and id {IDS['a blue square']};\neach guess becomes an e", col=1, row=7.25, h=135),
    dict(id="combine", kind="op", role="method", eyebrow="the product", label=f"e = e_empty + {GUIDE} [ ... ]",
         sub="(e_circle − e_empty)\n+ (e_square − e_empty)", col=2, row=7.25, h=135),
    dict(id="step", kind="op", role="plain", label="One reverse step", sub="ancestral (DDPM),\nlevel t to level t − 1", col=3, row=7.25, h=135),
    dict(id="corr", kind="op", role="method", eyebrow="optional", label="Langevin corrector",
         sub=f"{CS['main'][0]} or {CS['stronger'][0]} moves at each level,\neta = {ETA} (1 − abar_t)", col=3, row=8.55, h=135),
    dict(id="dec2", kind="decoder", role="off", label="VAE decoder", sub="the same decoder, frozen", col=4, row=7.25),
    dict(id="scorer", kind="service", role="plain", eyebrow="colour scorer", label="Count the shapes",
         sub=f"{SM['samples per model']} samples per model,\neach called one of\n{len(SC['categories'])} categories", col=5, row=7.25),
]

edges = [e for e in json.load(open(HERE.parents[3] / "3-understanding-compositionality/artifacts/scenes/how-the-caption-model-works/the-caption-model-high-level-design-explainer.json"))["edges"]]
for e in edges:
    if e.get("label") == "64 numbers":
        e["label"] = f"{D['caption table'][1]} numbers"
    if e.get("label") == "clean latent":
        e["label"] = "clean z0"
    if e.get("label") == "noisy latent":
        e["label"] = "noisy z_t"
    if e.get("label") == "3 noise\nestimates":
        e["label"] = "3 noise\nestimates e"
    if e["from"] == "trainset" and e["to"] == "pic":
        e["label"] = "each picture"
edges.append({"from": "capid", "to": "lookup", "style": "sync"}) if not any(e["from"] == "capid" for e in edges) else None

REL = TS["by relabelled count"]
spec = {
    "title": "The toy caption model: high-level design",
    "page_title": "Toy Caption Model",
    "subtitle": "Shares with SDXL: a VAE latent, caption conditioning, training with an empty caption, and guidance. It is not a U-Net and has no text encoder.",
    "lede": ("Read it top to bottom. Pictures with caption ids train a VAE that squeezes each picture into "
             f"{V['latent numbers']} numbers; a fully connected denoiser learns to guess the clean {V['latent numbers']} numbers from a noisy copy and a caption id; "
             "and sampling calls that one denoiser three times per step and adds the answers up as the product of the two single captions. "
             "Every picture on the diagram is real: drawn by the experiment's code, sampled by its saved runs, or recorded from one retrained model, as each card says."),
    "version_line": "",
    "look": "deep-learning",
    "box_w": 260, "box_h": 104, "col_w": 370, "row_h": 150,
    "static": True,
    "crumb": "The toy caption model",
    "lanes": [
        {"label": "1  TRAINING PICTURES AND CAPTIONS", "members": ["circles", "squares", "both", "relabel", "trainset"], "pad": 36},
        {"label": f"2  VAE: PICTURE TO {V['latent numbers']} NUMBERS AND BACK, {f(V['parameters'])} PARAMETERS, NO CAPTIONS", "members": ["pic", "enc", "latent", "dec", "recon"], "pad": 36},
        {"label": f"3  DENOISER: FULLY CONNECTED, {f(D['parameters'])} PARAMETERS, GUESSES THE CLEAN LATENT", "members": ["noise", "capid", "lookup", "mlp", "est", "loss"], "pad": 36},
        {"label": f"4  SAMPLING WITH THE PRODUCT: THE TRAINED DENOISER, {SM['denoiser calls per step for the product']} CALLS PER STEP, {SM['samples per model']} SAMPLES PER MODEL", "members": ["start", "calls", "combine", "step", "corr", "dec2", "scorer"], "pad": 36},
    ],
    "groups": [{"label": f"REPEATED {STEPS} TIMES, LEVEL {STEPS} DOWN TO 1", "members": ["calls", "combine", "step"]}],
    "nodes": nodes,
    "edges": edges,
}

src = "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/"
spec["details"] = [
    {"heading": "Where each box comes from", "lines": [
        f"All code paths are under <code>{src}</code>, at commit <code>db1325d7</code>; every number is read from <code>toy_relabelling_numbers.json</code> beside this page, written by <code>make_toy_relabelling_numbers.py</code>.",
        f"<strong>The three piles of pictures</strong>: <code>toy_sampler_hybrids.py</code> <code>draw</code>, <code>CIRCLES</code>, <code>SQUARES</code> ({PIX} pixels, a circle of {P['circle pixels']} pixels or a square of {P['square pixels']}); <code>toy_caption_dial_ldm.py</code> <code>PAIRS</code> (the {P['both-shapes layouts']} both-shapes layouts). {PER} per caption.",
        f"<strong>Relabel k pictures</strong>: <code>corpus_k</code> in <code>toy_caption_threshold.py</code>. Exactly k of the {PER} both-shapes pictures are added again under id {IDS['a red circle']} and again under id {IDS['a blue square']}, so the set grows from {f(k0)} to {f(k0)} + 2k pictures. Counts run: {', '.join(f(int(k)) for k in REL)}; three trainings each.",
        f"<strong>VAE</strong>: <code>class VAE</code> in <code>toy_caption_dial_ldm.py</code>, {f(V['parameters'])} parameters, latent {LAT} = {V['latent numbers']} numbers. Trained {f(V['training steps'])} steps, batch {V['batch']}, learning rate {g(V['learning rate'])}, KL weight {g(V['kl weight'])}; restarted up to {V['restarts allowed']} times until reconstruction error is at most {g(V['accept if reconstruction mse at most'])}.",
        f"<strong>Add noise at level t</strong> and <strong>Squared error</strong>: <code>train_denoiser</code>. z_t = sqrt(abar_t) z0 + sqrt(1 − abar_t) noise, {S['kind']} schedule with offset {g(S['offset s'])} over {STEPS} levels. The loss compares the output with the clean latent, so the denoiser guesses z0, never the noise. {f(D['training steps'])} steps, batch {D['batch']}, learning rate {g(D['learning rate'])}.",
        f"<strong>Caption id</strong>: ids {', '.join(f'{v} {k!r}' for k, v in IDS.items())}. During training the id is replaced by {IDS['empty']}, the empty caption, on {EMPTY_PCT}% of draws (<code>train_denoiser</code>), which is what makes guidance possible.",
        f"<strong>Caption vector</strong>: <code>nn.Embedding</code>, a learned {CAPT} table. No text is read anywhere in the model.",
        f"<strong>Denoiser</strong>: <code>class Denoiser</code>, {D['kind']} of {f(D['hidden width'])} with SiLU. Input: {V['latent numbers']} noisy latent numbers, {D['time features']} time features and the {D['caption table'][1]}-number caption row, {D['input width']} in all; {D['output width']} out. {f(D['parameters'])} parameters, caption table included.",
        f"<strong>3 denoiser calls and the product</strong>: <code>draw</code>. Each call's guess z0_hat becomes a noise estimate, {SM['noise estimate from predicted latent']}; the product is {SM['product rule']} with g = {GUIDE}; then one ancestral step.",
        f"<strong>Langevin corrector</strong>: <code>toy_caption_corrector.py</code>. After each step: {C['update']}. {CS['main'][0]} moves per level is the main setting, {CS['stronger'][0]} the stronger one.",
        f"<strong>Count the shapes</strong>: <code>toy_blob_scorer.py</code> <code>read</code>. Each pixel goes to the nearest of white, red, blue and purple, or to none when further than {g(SC['max colour distance for a pixel to count'])}. Categories: {'; '.join(SC['categories'])}. \"{SC['categories'][0]}\" means at least {SC['min pixels per colour']} red and {SC['min pixels per colour']} blue pixels, centres at least {g(SC['min centre gap along x (pixels)'])} pixels apart in x, and fewer than {SC['min pixels per colour']} purple pixels. A count such as 0 of {SM['samples per model']} is how many of a model's {SM['samples per model']} product samples fall in that category.",
    ]},
    {"heading": "How to read the lines and colours", "lines": [
        "Solid arrows carry data the next part waits on. The dotted two-headed arrow is the Langevin corrector, which runs only when switched on. The denoiser in lane 4 is the one trained in lane 3, called three times per step.",
        "Outline colours follow the ladder's key: blue for inputs, green for the hidden parts, red for what a network puts out, orange for the method's own parts (the product and the corrector), grey for a frozen part. Colour never fills a shape, because the toy's own pictures are red circles and blue squares.",
        f"Grey squares in a latent picture are its numbers, black for low and white for high, laid out as its {V['latent shape'][0]} channels of {V['latent shape'][1]} x {V['latent shape'][2]}.",
    ]},
    {"heading": "What is not drawn", "lines": [
        f"The one-caption sampler (e = e_empty + {GUIDE} (e_caption − e_empty)) is the same loop with two calls; it is described here, not drawn.",
        "The older relabelling by a random share (<code>corpus</code>, the share from 0 to 1) appears only in the corrector pictures, which come from those models; the counts experiment uses exact counts k.",
        "The template-library scorer in <code>toy_sampler_hybrids.py</code> is not what the counts use, so only the colour scorer is drawn.",
        "The VAE's KL term and log-variance clamp sit inside the VAE boxes and are not drawn as parts.",
    ]},
]

# ---- rung 0 additions: row positions with room for the picture cards, colour roles, and the cards
POS = {"both": (0, .85), "relabel": (1, .85), "circles": (1, 1.7), "squares": (1, 2.55), "trainset": (2, 1.7),
       "pic": (0, 3.95), "enc": (1, 3.95), "latent": (2, 3.95), "dec": (3, 3.95), "recon": (4, 3.95),
       "noise": (2, 5.75), "mlp": (3, 5.75), "est": (4, 5.75), "loss": (5, 5.75), "capid": (2, 6.65), "lookup": (3, 6.65),
       "start": (0, 8.75), "calls": (1, 8.75), "combine": (2, 8.75), "step": (3, 8.75), "dec2": (4, 8.75), "scorer": (5, 8.75),
       "corr": (3, 10.05)}
shift = lambda r: r + (0.5 if r >= 4.5 else 0) + (0.25 if r >= 7.5 else 0)
for n in nodes:
    n["col"], n["row"] = POS[n["id"]][0], shift(POS[n["id"]][1])
MAN = json.load(open(HERE / "the-toy-caption-model-pictures" / "manifest.json"))
PICS = json.load(open(HERE / "the-toy-caption-model-pictures" / "pictures.json"))["boxes"]
RT = MAN["_retrain"]
RTN = f"retrained model, {RT['relabelled']} relabelled, seed {RT['seed']}"
cat = lambda prefix, i: MAN[f"_{prefix}"]["categories"][i]
idname = {v: k for k, v in IDS.items()}
lvl = lambda name: str(int(name.rsplit("-k", 1)[1][:3]))

def lab(img, box):
    """The short label under one tile, read from the manifest or the numbers file."""
    s = img.rsplit("/", 1)[-1][:-4]
    if s.startswith("clean-circle-") or s.startswith("clean-square-"):
        if box == "capid":
            return [f"id {IDS['a red circle'] if 'circle' in s else IDS['a blue square']}"]
        return [f"x = {g(XP[s.rsplit('-', 1)[1]])}"]
    if s.startswith("clean-both-"):
        if box == "capid":
            return [f"id {IDS['a red circle and a blue square']}"]
        return ["circle left" if "circle-left" in s else "square left"]
    if s == "noisy-train-relabelled-source":
        return [f"id {IDS['a red circle and a blue square']}"]
    if s.startswith("noisy-train-"):
        return [f"id {MAN['_ids'][s[len('noisy-train-'):]]}"]
    if s.startswith("noised-k"):
        return [f"t = {lvl(s)}"]
    if s.startswith("guess-k100-"):
        c = {"empty": IDS["empty"], "circle": IDS["a red circle"], "square": IDS["a blue square"], "both": IDS["a red circle and a blue square"]}[s.rsplit("-", 1)[1]]
        return [f"id {c}" + (" (empty)" if c == IDS["empty"] else "")]
    if s == "guess-latent-k100-both":
        return ["guess z0_hat"]
    if s == "clean-latent-target":
        return ["target z0"]
    if s.startswith("poe-call-k100-"):
        w = s.rsplit("-", 1)[1]
        return ["empty" if w == "empty" else f"id {IDS['a red circle'] if w == 'circle' else IDS['a blue square']}"]
    if s == "poe-combined-k100":
        return ["from e"]
    if s.startswith("poe-run-k"):
        return [f"t = {lvl(s)}"]
    if s == "caption-table":
        return [f"rows: id {', '.join(str(v) for v in IDS.values())}, top to bottom"]
    for prefix, head in (("share0-poe-corrector20", f"share 0, {CS['stronger'][0]} moves"), ("share1-poe-corrector20", f"share 1, {CS['stronger'][0]} moves"),
                         ("share0-poe", "share 0, plain"), ("share1-poe", "share 1, plain"),
                         ("product-k0-s20261023", "0 relabelled\nseed 20261023"), ("product-k0-s20261022", "0 relabelled\nseed 20261022")):
        if s.startswith(prefix + "-"):
            return head.split("\n") + [cat(prefix, int(s.rsplit("-", 1)[1]))]
    if s.startswith("scorer-map-"):
        return ["each pixel's colour"]
    return [s]

TITLES = {
    "both": f"the {P['both-shapes layouts']} layouts", "circles": f"the {P['single-shape pictures per caption']} positions",
    "squares": f"the {P['single-shape pictures per caption']} positions", "relabel": "one drawn picture under three ids",
    "trainset": f"copies with pixel noise, sd {g(P['pixel noise sd added to every training copy'])}",
    "noise": "z_t at five levels, decoded to be seen", "capid": "what each id names",
    "lookup": f"the trained table", "mlp": f"its guess at t = 100 under each id",
    "loss": "what the loss compares, t = 100", "calls": "the three guesses at t = 100, one run",
    "combine": "what the product's e implies", "step": f"the product's guess along one run",
    "corr": "saved samples of the share models", "dec2": "finished samples of saved runs",
    "scorer": "a saved sample and the colour the scorer gives each pixel",
}
CARD_AT = {"both": (0, .12), "relabel": (2, .85), "circles": (0, 1.7), "squares": (0, 2.55), "trainset": (3.05, 1.7),
           "mlp": (3, 4.9), "loss": (5, 4.9), "noise": (0.55, 5.75), "capid": (0.75, 6.65), "lookup": (4.45, 6.65),
           "step": (2.5, 7.85), "dec2": (3.95, 7.85), "scorer": (5.35, 7.85), "calls": (1, 9.55), "combine": (2, 9.55),
           "corr": (4.6, 10.05)}
INLINE = {"pic", "latent", "recon", "est", "start"}
INLINE_NOTE = {"pic": "shown: one training copy", "latent": "shown: retrained model", "recon": "shown: retrained model",
               "est": "shown: guess at t = 100, retrained", "start": "shown: one standard normal draw"}
cards = []
RETRAINED = ("vae-", "noised-", "guess-", "poe-", "start-", "caption-table", "time-features", "clean-latent-target")
SAVED = ("product-", "share", "circle-k", "square-k", "scorer-map-")
for n in nodes:
    face = PICS.get(n["id"], {}).get("face", {})
    items = PICS.get(n["id"], {}).get(face.get("side", ""), [])[:face.get("count", 0)]
    if not items:
        continue
    if n["id"] in INLINE:
        n["picture"] = items[0]["image"]
        n["sub"] = n["sub"] + "\n" + INLINE_NOTE[n["id"]]
        continue
    tiles = [{"image": it["image"], "label": lab(it["image"], n["id"])} for it in items]
    wide = n["id"] in ("lookup",)
    names = [it["image"].rsplit("/", 1)[-1] for it in items]
    src = ("retrained" if all(x.startswith(RETRAINED) for x in names) else
           "saved runs" if all(x.startswith(SAVED) for x in names) else "drawn by the code")
    cards.append({"box": n["id"], "at": (CARD_AT[n["id"]][0], shift(CARD_AT[n["id"]][1])), "title": TITLES[n["id"]], "source": src, "tiles": tiles, "wide": wide})
spec["cards"] = cards
spec["source_key"] = f"Picture tags: drawn by the code = the experiment's own draw() and corpus_k; saved runs = samples saved by the experiment's own runs; retrained = one model retrained here with the same code ({RT['relabelled']} relabelled, seed {RT['seed']}, VAE error {g(RT['vae mse'])}), not one of the reported trainings."
spec["roles"] = {n["id"]: n.pop("role") for n in nodes}
spec["version_line"] = f"Version 2, 2026-11-18: {len(nodes)} boxes, {len(edges)} arrows, {len(cards)} picture cards and {len(INLINE)} pictures in place of tensor grids. Grown from version 1 of 2026-11-05 in subproject 3."
if "--pictures" in sys.argv:
    spec["pictures_file"] = sys.argv[sys.argv.index("--pictures") + 1]
json.dump(spec, open(HERE / f"{STEM}.json", "w"), indent=1, ensure_ascii=False)
print("wrote", HERE / f"{STEM}.json", len(nodes), "boxes", len(edges), "arrows")
