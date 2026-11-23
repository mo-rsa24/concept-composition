"""Read only the safetensors headers and the projection tensors of two public checkpoints, by HTTP range, and compare them."""
import json, struct, urllib.request
import numpy as np
U = {"noisyclip": "https://huggingface.co/asiimo/noisyclip/resolve/main/model.safetensors",
     "base": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors"}
def rng(url, a, b):
    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "python-urllib"})
    return urllib.request.urlopen(r, timeout=120).read()
def header(url):
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    return json.loads(rng(url, 8, 7 + n)), 8 + n
def tensor(url, h, base, k):
    m = h[k]; a, b = m["data_offsets"]
    dt = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}[m["dtype"]]
    return np.frombuffer(rng(url, base + a, base + b - 1), dtype=dt).reshape(m["shape"]).astype(np.float64), m["dtype"]
H = {n: header(u) for n, u in U.items()}
for n, (h, b) in H.items():
    print(n, "tensors", len([k for k in h if k != "__metadata__"]), "meta", h.get("__metadata__"))
for k in ["visual_projection.weight", "text_projection.weight", "logit_scale", "text_model.encoder.layers.0.self_attn.k_proj.weight", "vision_model.encoder.layers.0.self_attn.k_proj.weight"]:
    if k not in H["noisyclip"][0] or k not in H["base"][0]:
        print(k, "missing", k in H["noisyclip"][0], k in H["base"][0]); continue
    x, dx = tensor(U["noisyclip"], H["noisyclip"][0], H["noisyclip"][1], k)
    y, dy = tensor(U["base"], H["base"][0], H["base"][1], k)
    d = x - y
    print(f"{k}: shape {x.shape} dtype {dx}/{dy} max|diff| {np.abs(d).max():.3e} rel_norm {np.linalg.norm(d)/max(np.linalg.norm(y),1e-12):.3e}", "values" if x.size==1 else "", (x.item(), y.item()) if x.size==1 else "")
