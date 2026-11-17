"""Train guided-diffusion's 64 px model on our cat / dog images with a null label (plan 20, piece 2).

Runs their TrainLoop, model and diffusion unchanged. Three things are replaced around them:
  * mpi4py and blobfile, which co3 / co3_bw lack, by one-process stand-ins put in sys.modules
    before guided_diffusion is imported;
  * their setup_dist, which would reset CUDA_VISIBLE_DEVICES from the MPI rank, by a
    one-process group started here, so the card pinned at launch is the card used;
  * their load_data, by a generator over our image folder that carries the label dropout.

Labels: cat 0, dog 1, null 2. A both-animal image (--arm all) takes cat or dog at random each
time it is drawn. With probability --p-null the label is replaced by null, so one network gives
the cat expert, the dog expert and the null expert the product needs.
"""
import argparse
import os
import socket
import sys
import time
import types

# --- one-process stand-ins, installed before guided_diffusion loads -------------------------
mpi4py = types.ModuleType("mpi4py")


class _Comm:
    rank, size = 0, 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, x, root=0):
        return x


mpi4py.MPI = types.SimpleNamespace(COMM_WORLD=_Comm())
sys.modules["mpi4py"] = mpi4py
sys.modules["mpi4py.MPI"] = mpi4py.MPI

blobfile = types.ModuleType("blobfile")
blobfile.BlobFile = open
blobfile.join = os.path.join
blobfile.dirname = os.path.dirname
blobfile.basename = os.path.basename
blobfile.exists = os.path.exists
blobfile.listdir = os.listdir
blobfile.isdir = os.path.isdir
sys.modules["blobfile"] = blobfile

GD = "/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion"
sys.path.insert(0, GD)

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
import torch.distributed as dist  # noqa: E402
from PIL import Image  # noqa: E402

from guided_diffusion import logger, script_util  # noqa: E402
from guided_diffusion.resample import create_named_schedule_sampler  # noqa: E402
from guided_diffusion.train_util import TrainLoop  # noqa: E402

CAT, DOG, NULL = 0, 1, 2
IMAGES = "/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/images/64"

# The README's 64 px ImageNet model flags, with three labels instead of 1,000.
MODEL_64 = dict(
    image_size=64, num_channels=192, num_res_blocks=3, num_heads=4, num_heads_upsample=-1,
    num_head_channels=64, attention_resolutions="32,16,8", channel_mult="", dropout=0.1,
    class_cond=True, use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True,
    use_fp16=True, use_new_attention_order=True, learn_sigma=True, diffusion_steps=1000,
    noise_schedule="cosine", timestep_respacing="", use_kl=False, predict_xstart=False,
    rescale_timesteps=False, rescale_learned_sigmas=False,
)


def start_one_process_group():
    s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1")
    dist.init_process_group(backend="nccl", init_method="env://")


def load_images(arm):
    names = sorted(os.listdir(IMAGES))
    keep = [n for n in names if n.split("_")[0] in ("cat", "dog") or (arm == "all" and n.startswith("both_"))]
    arrs = np.stack([np.asarray(Image.open(os.path.join(IMAGES, n)).convert("RGB")) for n in keep])
    kinds = np.array([n.split("_")[0] for n in keep])
    return arrs, kinds


def batches(arrs, kinds, batch_size, p_null, seed):
    """Yield (images in [-1, 1], {"y": labels}) forever, with the null label at rate p_null."""
    rng = np.random.default_rng(seed)
    x_all = th.from_numpy(arrs).permute(0, 3, 1, 2).float() / 127.5 - 1
    while True:
        idx = rng.integers(0, len(arrs), size=batch_size)
        x = x_all[idx].clone()
        flip = th.from_numpy(rng.random(batch_size) < 0.5)
        x[flip] = x[flip].flip(-1)
        y = np.where(kinds[idx] == "cat", CAT, DOG)
        both = kinds[idx] == "both"
        y[both] = rng.integers(0, 2, size=both.sum())  # cat or dog at random per draw
        y[rng.random(batch_size) < p_null] = NULL  # the label dropout
        yield x, {"y": th.from_numpy(y).long()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["no_both", "all"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--p-null", type=float, default=0.1)
    p.add_argument("--num-channels", type=int, default=192)
    p.add_argument("--save-interval", type=int, default=5000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--wandb-name", default="")
    args = p.parse_args()

    assert th.cuda.is_available(), "no CUDA device: on a Blackwell use co3_bw"
    th.manual_seed(args.seed)
    start_one_process_group()
    os.makedirs(args.run_dir, exist_ok=True)
    logger.configure(dir=args.run_dir, format_strs=["stdout", "log", "csv"])
    logger.log(f"node {socket.gethostname()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
               f"pid {os.getpid()} device {th.cuda.get_device_name(0)} args {vars(args)}")

    if args.wandb_name:
        import wandb
        wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=args.wandb_name,
                   config={**vars(args), **MODEL_64, "num_classes": 3, "labels": "cat 0, dog 1, null 2"})
        _dump = logger.dumpkvs

        def dump_and_log():
            kvs = dict(logger.getkvs())
            out = _dump()
            wandb.log({k: v for k, v in kvs.items() if isinstance(v, (int, float))}, step=int(kvs.get("step", 0)))
            return out

        logger.dumpkvs = dump_and_log
        import guided_diffusion.train_util as tu
        tu.logger.dumpkvs = dump_and_log

    script_util.NUM_CLASSES = 3  # cat, dog, null
    flags = dict(MODEL_64, num_channels=args.num_channels)
    model, diffusion = script_util.create_model_and_diffusion(**flags)
    model.to("cuda")
    logger.log(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    arrs, kinds = load_images(args.arm)
    counts = {k: int((kinds == k).sum()) for k in ("cat", "dog", "both")}
    logger.log(f"arm {args.arm}: {len(arrs)} images {counts}, null rate {args.p_null}")

    t0 = time.time()
    TrainLoop(
        model=model, diffusion=diffusion, data=batches(arrs, kinds, args.batch_size, args.p_null, args.seed),
        batch_size=args.batch_size, microbatch=-1, lr=args.lr, ema_rate="0.9999",
        log_interval=args.log_interval, save_interval=args.save_interval, resume_checkpoint=args.resume,
        use_fp16=True, fp16_scale_growth=1e-3,
        schedule_sampler=create_named_schedule_sampler("uniform", diffusion),
        weight_decay=0.0, lr_anneal_steps=args.steps,
    ).run_loop()
    logger.log(f"DONE {args.steps} steps in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
