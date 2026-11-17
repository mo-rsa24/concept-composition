import sys, numpy as np, torch as th, torch.nn.functional as F
sys.path.insert(0, "/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/code")
from classifiers import released_classifier, imagenet_classifier, imagenet_cat_dog_logp
D = "/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos"
u = np.load(f"{D}/draws/product_256_a.npz")["u8"]
x = th.from_numpy(u).float().cuda() / 127.5 - 1
sep, _ = released_classifier(256, 2); sep.load_state_dict(th.load(f"{D}/classifiers/separate_256.pt", weights_only=True)); sep.cuda().eval()
inet = imagenet_classifier().cuda().eval()
ps, pi = [], []
with th.no_grad():
    for i in range(0, len(x), 16):
        b = x[i:i+16]; t = th.zeros(len(b), dtype=th.long, device="cuda")
        ps.append(th.sigmoid(sep(b, t)).cpu()); pi.append(imagenet_cat_dog_logp(inet, b, t).exp().cpu())
ps, pi = th.cat(ps), th.cat(pi)
n = len(ps)
print(f"{n} old product draws (mostly food and objects by eye)")
print(f"our classifier:      says cat on {(ps[:,0]>0.5).sum()} of {n}, dog on {(ps[:,1]>0.5).sum()}, cat or dog on {((ps>0.5).any(1)).sum()}; mean p(cat) {ps[:,0].mean():.2f}, p(dog) {ps[:,1].mean():.2f}")
print(f"released classifier: says cat on {(pi[:,0]>0.5).sum()} of {n}, dog on {(pi[:,1]>0.5).sum()}, cat or dog on {((pi>0.5).any(1)).sum()}; mean p(cat) {pi[:,0].mean():.2f}, p(dog) {pi[:,1].mean():.2f}")
