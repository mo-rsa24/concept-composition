"""Every number the CO3 ladder prints, computed from the authors' code, the SDXL scheduler config and the
project's released settings. Writes co3_numbers.json beside this file.

diffusers is not installed on this laptop, so the DDIM schedule is recomputed in numpy exactly as
diffusers' DDIMScheduler builds it from `source/sdxl-scheduler-config.json` (scaled_linear betas from
beta_start to beta_end over num_train_timesteps, square-root spacing; "leading" timesteps plus steps_offset;
final_alpha_cumprod = alphas_cumprod[0] because set_alpha_to_one is false).

Run:  python3 co3_numbers.py
"""
import ast
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "source"
CO3 = Path("/home/molef/PhD/replications/co3")
CO3_PY = CO3 / "composers" / "Co3.py"
ATTN_PY = CO3 / "composers" / "utils" / "attn_utils.py"


def line(path, n, must_contain):
    """Line n (1-based) of path, asserted to hold the expected text so a moved line stops the build."""
    text = path.read_text().splitlines()[n - 1]
    assert must_contain in text, f"{path.name}:{n} no longer holds {must_contain!r}: {text!r}"
    return text


def released_settings():
    """RELEASED_SETTINGS from the project's co3_common.py, parsed, never typed."""
    tree = ast.parse((SRC / "co3_common.py.copy").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "RELEASED_SETTINGS":
            return {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    raise RuntimeError("RELEASED_SETTINGS not found")


def schedule(cfg, n_steps):
    betas = np.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, cfg["num_train_timesteps"], dtype=np.float32) ** 2
    alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
    final_alpha_cumprod = 1.0 if cfg["set_alpha_to_one"] else float(alphas_cumprod[0])
    assert cfg["timestep_spacing"] == "leading"
    step_ratio = cfg["num_train_timesteps"] // n_steps
    timesteps = (np.arange(0, n_steps) * step_ratio).round()[::-1].astype(np.int64) + cfg["steps_offset"]
    return betas, alphas_cumprod, final_alpha_cumprod, timesteps


def main():
    cfg = json.loads((SRC / "sdxl-scheduler-config.json").read_text())
    s = released_settings()
    n_steps = s["n_timesteps"]
    betas, ac, final_ac, timesteps = schedule(cfg, n_steps)

    # Co3.py line 89: skip = N_ts // n_timesteps, N_ts = 1000 timesteps before set_timesteps.
    line(CO3_PY, 89, "self.skip = N_ts // self.config.n_timesteps")
    skip = cfg["num_train_timesteps"] // n_steps
    # Co3.py line 91 prepends 1.0, so CO3's alpha(t) is the scheduler's alphas_cumprod[t - 1].
    line(CO3_PY, 91, "torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])")
    line(CO3_PY, 201, "self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod")
    ac_co3 = np.concatenate([[1.0], ac])

    def alpha_co3(t):
        return float(ac_co3[t]) if t >= 0 else final_ac

    # The hybrid branch, read off its lines.
    line(CO3_PY, 103, "self.scheduler.timesteps[:t_cond]")
    line(CO3_PY, 306, "beta = 1.0 if t == self.start_t else self.config.beta")
    line(CO3_PY, 307, "num_corrector_steps = 10 if t >= self.scheduler.timesteps[1] else self.config.num_latent_corrector_steps")
    line(CO3_PY, 309, "if t >= self.scheduler.timesteps[self.config.num_resampling_steps-1]:")
    line(CO3_PY, 361, "if t == 1:")
    beta_first = float(re.search(r"beta = ([0-9.]+) if", line(CO3_PY, 306, "beta =")).group(1))
    p_early = int(re.search(r"= (\d+) if t >=", line(CO3_PY, 307, "num_corrector_steps")).group(1))
    corrected = [int(t) for t in timesteps[: s["num_ts_to_correct"]]]
    resample_from = int(timesteps[s["num_resampling_steps"] - 1])
    ten_from = int(timesteps[1])

    # Batch sizes: the corrector stacks x for [uncond, C, c_1..c_K] (attn_utils.py 182 and 238); the main step stacks 2.
    line(ATTN_PY, 182, "torch.cat([x] * (num_concepts + 2))")
    line(CO3_PY, 338, "torch.cat([x]+[x])")
    log = (SRC / "run-log-excerpt-l6-m1-p2-93468.txt").read_text()
    prompts_line = next(l for l in log.splitlines() if l.startswith("Processed prompts"))
    prompts = ast.literal_eval(prompts_line.split("prompts ", 1)[1].split(", prompts_single")[0])
    # Co3.py 280 drops the last text embedding (the repeated full prompt) before the corrector.
    line(CO3_PY, 280, "text_embed_cond[:-1]")
    corrector_conditions = ["(empty prompt)"] + prompts[:-1]
    K = len(corrector_conditions) - 2

    steps = []
    for i, t in enumerate(int(x) for x in timesteps):
        next_t = t - skip
        at, at_next = alpha_co3(t), alpha_co3(next_t)
        row = dict(step=i, t=t, next_t=next_t, alpha_bar=at, alpha_bar_next=at_next,
                   sqrt_alpha_bar=at ** 0.5, sqrt_one_minus_alpha_bar=(1 - at) ** 0.5,
                   alpha_bar_without_prepend=float(ac[t]), co3=t in corrected)
        if row["co3"]:
            row["algorithm"] = "resampler" if t >= resample_from else "corrector"
            row["iterations"] = p_early if t >= ten_from else s["num_latent_corrector_steps"]
            row["beta"] = beta_first if t == int(timesteps[0]) else s["beta"]
            row["unet_images"] = row["iterations"] * (K + 2) + 2
        else:
            row.update(algorithm="plain", iterations=0, beta=None, unet_images=2)
        row["last_step_returns_x0_hat"] = t == 1
        steps.append(row)

    # Check the schedule against what the real run printed.
    printed = [(m.group(1), int(m.group(2))) for m in re.finditer(r"(resampling|correction) step at t=(\d+)", log)]
    computed = [("resampling" if r["algorithm"] == "resampler" else "correction", r["t"]) for r in steps if r["co3"]]
    assert printed == computed, (printed, computed)

    # Weights. Uniform start (attn_utils 176/232), anchors w0 (177, 233), equation 15 (155).
    line(ATTN_PY, 176, "concept_weights = [-1.0/num_concepts]*num_concepts")
    w0_res = float(re.search(r"\[([0-9.]+)\] \+ concept_weights", line(ATTN_PY, 177, "composition_weights = [1.0]")).group(1))
    w0_cor = float(re.search(r"\[([0-9.]+)\] \+ concept_weights", line(ATTN_PY, 233, "composition_weights = [2.0]")).group(1))
    line(ATTN_PY, 145, "a = torch.exp(-beta * dists)")
    line(ATTN_PY, 155, "w = -a / a.sum()")

    def eq15(d, beta):
        a = np.exp(-beta * np.asarray(d, dtype=np.float64))
        return -a / a.sum()

    gaps = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    curve_gaps = np.round(np.linspace(-6, 6, 121), 3)
    betas_used = sorted({beta_first, s["beta"]})
    weights = dict(
        K=K,
        uniform_concept_weight=-1.0 / K,
        w0_resampler=w0_res, w0_corrector=w0_cor,
        sum_resampler=w0_res + K * (-1.0 / K), sum_corrector=w0_cor + K * (-1.0 / K),
        betas=betas_used,
        # w_1 for K = 2 when d_1 - d_2 = gap; w_2 = -1 - w_1.
        w1_at_gap={str(b): {str(g): float(eq15([g, 0.0], b)[0]) for g in gaps} for b in betas_used},
        curve={"gap": curve_gaps.tolist(),
               **{str(b): [float(eq15([g, 0.0], b)[0]) for g in curve_gaps] for b in betas_used}},
        latent_numbers=4 * (s["resolution_h"] // 8) * (s["resolution_w"] // 8),
    )

    first = steps[0]
    out = dict(
        about="Every number the CO3 ladder prints. Written by co3_numbers.py; never edit by hand.",
        code=dict(checkout=str(CO3), commit=subprocess.check_output(["git", "-C", str(CO3), "rev-parse", "HEAD"], text=True).strip()),
        schedule_source=dict(config="source/sdxl-scheduler-config.json",
                             computed_with="numpy reimplementation of diffusers DDIMScheduler (diffusers not installed here)",
                             **{k: cfg[k] for k in ("beta_start", "beta_end", "beta_schedule", "num_train_timesteps",
                                                   "set_alpha_to_one", "steps_offset", "timestep_spacing", "prediction_type")}),
        settings=s,
        skip=skip,
        final_alpha_cumprod=final_ac,
        timesteps=[int(t) for t in timesteps],
        co3_timesteps=corrected,
        resample_from_t=resample_from,
        ten_iterations_from_t=ten_from,
        beta_first_step=beta_first,
        iterations_early=p_early,
        corrector_conditions=corrector_conditions,
        concepts=prompts[1:-1],
        joint_prompt=prompts[0],
        steps=steps,
        totals=dict(
            co3_iterations=sum(r["iterations"] for r in steps),
            resampler_iterations=sum(r["iterations"] for r in steps if r["algorithm"] == "resampler"),
            corrector_iterations=sum(r["iterations"] for r in steps if r["algorithm"] == "corrector"),
            unet_images_co3=sum(r["unet_images"] for r in steps),
            unet_images_plain_cfg=2 * n_steps,
        ),
        first_step=dict(t=first["t"], alpha_bar=first["alpha_bar"], sqrt_one_minus_alpha_bar=first["sqrt_one_minus_alpha_bar"],
                        alpha_bar_without_prepend=first["alpha_bar_without_prepend"],
                        relative_shift=(first["alpha_bar"] - first["alpha_bar_without_prepend"]) / first["alpha_bar_without_prepend"]),
        weights=weights,
        run_log_check=dict(file="source/run-log-excerpt-l6-m1-p2-93468.txt", printed=printed, matches=True),
        render=json.loads((SRC / "co3-eagle-fox-seed-6001.json").read_text()),
        paper_settings=dict(  # cited, with pages, from the PDF on the shelf (arXiv 2509.25940v2)
            guidance_scale=dict(value=5.0, page=6), corrected_steps=dict(value=10, page=17),
            resampling_steps=dict(value=3, page=17), iterations=dict(value=5, page=17), beta=dict(value=0.8, page=6)),
    )
    (HERE / "co3_numbers.json").write_text(json.dumps(out, indent=1))
    print(f"wrote co3_numbers.json: {n_steps} steps, CO3 at t={corrected}, K={K}, "
          f"alpha_bar(981)={first['alpha_bar']:.6f}, totals={out['totals']}")


if __name__ == "__main__":
    main()
