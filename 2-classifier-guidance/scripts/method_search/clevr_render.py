"""Render CLEVR scenes with each object placed at a chosen image position (step M7).

Runs inside Blender 2.78c, whose Python is 3.5. It calls the released generator's render_scene unchanged and replaces only the step
that picks where objects go. Shapes, sizes, colours, materials, camera jitter and light jitter stay the generator's defaults.

A target (x, y) is a fraction of the image width and height, measured from the top left. Each object's target gets a uniform offset of up
to OFFSET, and the object goes on the ground point that projects there under this image's jittered camera. Each image has its own seed,
and an image whose scene file exists is skipped, so a job can resume.

Stage 2 of the CLEVR plan uses --colours in place of --targets: one object per listed colour, placed anywhere by the generator's own
distance and margin rules, so no position is fixed.

Usage:
  CLEVR_GEN=<image_generation folder> blender --background --threads N --python clevr_render.py -- \
      --name p1 --targets "0.30,0.30" --seed-base 1000000 --start 0 --count 8 --out <dir>
  ... --name r2 --colours red,red --seed-base 6200000 --start 0 --count 8 --out <dir>
"""
from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time

sys.dont_write_bytecode = True      # the generator's folder belongs to another session and stays unwritten
GEN = os.environ["CLEVR_GEN"]
sys.path.insert(0, GEN)
os.chdir(GEN)

import bpy  # noqa: E402
import bpy_extras  # noqa: E402
from mathutils import Vector  # noqa: E402
import utils  # noqa: E402
import render_images as generator  # noqa: E402

OFFSET = 0.02
MAX_TRIES = 50
TARGETS = []
COLOURS = []


class ClosingTempfile:
    """The generator's visibility check never closes its temp file, which ends a run near image 1,000 at the 1,024 open-file limit."""

    @staticmethod
    def mkstemp(*args, **kwargs):
        handle, path = tempfile.mkstemp(*args, **kwargs)
        os.close(handle)
        return handle, path


generator.tempfile = ClosingTempfile


def project(camera, x, y, z):
    point = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera, Vector((x, y, z)))
    return point.x, 1.0 - point.y


def ground_point(camera, target_x, target_y, z):
    """Newton's method on the camera projection: the (x, y) on the ground whose point at height z lands on the target."""
    x, y = 0.0, 0.0
    step = 1e-3
    for _ in range(40):
        px, py = project(camera, x, y, z)
        ex, ey = px - target_x, py - target_y
        if max(abs(ex), abs(ey)) < 1e-4:
            return x, y
        ax, ay = project(camera, x + step, y, z)
        bx, by = project(camera, x, y + step, z)
        a, b, c, d = (ax - px) / step, (bx - px) / step, (ay - py) / step, (by - py) / step
        det = a * d - b * c
        x -= (d * ex - b * ey) / det
        y -= (a * ey - c * ex) / det
    raise RuntimeError("no ground point for target (%.3f, %.3f)" % (target_x, target_y))


def place_objects(scene_struct, num_objects, args, camera):
    """Stands in for the generator's add_random_objects, with the same attribute draws and fixed positions."""
    with open(args.properties_json, "r") as f:
        properties = json.load(f)
    colours = {name: [float(c) / 255.0 for c in rgb] + [1.0] for name, rgb in properties["colors"].items()}
    materials = [(v, k) for k, v in properties["materials"].items()]
    shapes = [(v, k) for k, v in properties["shapes"].items()]
    sizes = list(properties["sizes"].items())
    bpy.context.scene.update()
    scene_struct["targets"] = [list(t) for t in TARGETS]

    for attempt in range(MAX_TRIES):
        placed, objects, blender_objects = [], [], []
        fits = True
        for target_x, target_y in TARGETS:
            size_name, r = random.choice(sizes)
            obj_name, obj_name_out = random.choice(shapes)
            if obj_name == "Cube":
                r /= math.sqrt(2)
            aim_x = target_x + random.uniform(-OFFSET, OFFSET)
            aim_y = target_y + random.uniform(-OFFSET, OFFSET)
            x, y = ground_point(camera, aim_x, aim_y, r)
            if any(math.sqrt((x - xx) ** 2 + (y - yy) ** 2) - r - rr < args.min_dist for xx, yy, rr in placed):
                fits = False
                break
            color_name, rgba = random.choice(list(colours.items()))
            theta = 360.0 * random.random()
            utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
            obj = bpy.context.object
            blender_objects.append(obj)
            placed.append((x, y, r))
            mat_name, mat_name_out = random.choice(materials)
            utils.add_material(mat_name, Color=rgba)
            objects.append({"shape": obj_name_out, "size": size_name, "material": mat_name_out,
                            "3d_coords": tuple(obj.location), "rotation": theta,
                            "pixel_coords": utils.get_camera_coords(camera, obj.location), "color": color_name,
                            "aimed_at": [aim_x, aim_y]})
        if fits and (not blender_objects or generator.check_visibility(blender_objects, args.min_pixels_per_object)):
            return objects, blender_objects
        for obj in blender_objects:
            utils.delete_object(obj)
    raise RuntimeError("could not place the objects in %d tries" % MAX_TRIES)


def colour_objects(scene_struct, num_objects, args, camera):
    """Stage 2: one object per entry of COLOURS, anywhere on the ground, by the generator's own rules for distance and margin.

    Shapes, sizes and materials are the generator's random draws. Positions are uniform in (-3, 3), each object keeps the generator's
    minimum distance and its margin along the four directions to every other, and the visibility check rejects a hidden object.
    """
    with open(args.properties_json, "r") as f:
        properties = json.load(f)
    colours = {name: [float(c) / 255.0 for c in rgb] + [1.0] for name, rgb in properties["colors"].items()}
    materials = [(v, k) for k, v in properties["materials"].items()]
    shapes = [(v, k) for k, v in properties["shapes"].items()]
    sizes = list(properties["sizes"].items())
    bpy.context.scene.update()
    scene_struct["asked_colours"] = list(COLOURS)
    directions = [scene_struct["directions"][d] for d in ("left", "right", "front", "behind")]

    for attempt in range(MAX_TRIES):
        placed, objects, blender_objects = [], [], []
        fits = True
        for colour_name in COLOURS:
            size_name, r = random.choice(sizes)
            obj_name, obj_name_out = random.choice(shapes)
            if obj_name == "Cube":
                r /= math.sqrt(2)
            for _ in range(args.max_retries):
                x, y = random.uniform(-3, 3), random.uniform(-3, 3)
                good = True
                for xx, yy, rr in placed:
                    dx, dy = x - xx, y - yy
                    if math.sqrt(dx * dx + dy * dy) - r - rr < args.min_dist:
                        good = False
                        break
                    if any(0 < dx * d[0] + dy * d[1] < args.margin for d in directions):
                        good = False
                        break
                if good:
                    break
            else:
                fits = False
                break
            theta = 360.0 * random.random()
            utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
            obj = bpy.context.object
            blender_objects.append(obj)
            placed.append((x, y, r))
            mat_name, mat_name_out = random.choice(materials)
            utils.add_material(mat_name, Color=colours[colour_name])
            objects.append({"shape": obj_name_out, "size": size_name, "material": mat_name_out,
                            "3d_coords": tuple(obj.location), "rotation": theta,
                            "pixel_coords": utils.get_camera_coords(camera, obj.location), "color": colour_name})
        if fits and (not blender_objects or generator.check_visibility(blender_objects, args.min_pixels_per_object)):
            return objects, blender_objects
        for obj in blender_objects:
            utils.delete_object(obj)
    raise RuntimeError("could not place the objects in %d tries" % MAX_TRIES)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Paper 1's joint mode: red and blue objects together, touching allowed, never interpenetrating, with per-object masks at 64 pixels.
# Settings: plans/paper-1-clevr-touching/plans/03-full-plan.md, "The data and the splits". Written for Blender 2.78c's Python 3.5.

JOINT = []            # the colour of each object, red ones first, set per scene
MASK_SIZE = 0         # 64: write each object's visible mask at 64 by 64; 0: none
MASK_PATH = [None]    # where the current scene's masks go, set before each scene


def _corners(x, y, half, theta):
    """The four corners, in the ground plane, of a square of half-side `half` centred at (x, y) and turned by theta degrees."""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((half, half), (-half, half), (-half, -half), (half, -half))]


def _point_square_distance(px, py, x, y, half, theta):
    """Distance in the ground plane from a point to a turned square's footprint (0 inside it)."""
    c, s = math.cos(math.radians(-theta)), math.sin(math.radians(-theta))
    lx, ly = c * (px - x) - s * (py - y), s * (px - x) + c * (py - y)
    dx, dy = max(abs(lx) - half, 0.0), max(abs(ly) - half, 0.0)
    return math.sqrt(dx * dx + dy * dy)


def _squares_overlap(a, b):
    """Separating-axis test for two turned squares (x, y, half, theta); touching edges do not overlap."""
    pa, pb = _corners(*a), _corners(*b)
    for poly in (pa, pb):
        for i in range(4):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            proj_a = [nx * px + ny * py for px, py in pa]
            proj_b = [nx * px + ny * py for px, py in pb]
            if max(proj_a) <= min(proj_b) + 1e-9 or max(proj_b) <= min(proj_a) + 1e-9:
                return False
    return True


def footprints_overlap(a, b):
    """True when two objects' ground footprints overlap, so the objects would pass through each other.

    Each object is (shape, x, y, r, theta): a sphere or cylinder covers a disc of radius r, a cube a square of half-side r turned by
    theta (r is the generator's size after its division by sqrt 2 for cubes, which add_object scales the 2-unit cube by). Every
    shape stands on the ground and is no wider above its footprint, so footprints that do not overlap mean bodies that do not.
    """
    sa, ax, ay, ar, at = a
    sb, bx, by, br, bt = b
    if sa != "Cube" and sb != "Cube":
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) < ar + br - 1e-9
    if sa == "Cube" and sb == "Cube":
        return _squares_overlap((ax, ay, ar, at), (bx, by, br, bt))
    if sa == "Cube":
        return _point_square_distance(bx, by, ax, ay, ar, at) < br - 1e-9
    return _point_square_distance(ax, ay, bx, by, br, bt) < ar - 1e-9


def joint_objects(scene_struct, num_objects, args, camera):
    """Paper 1's joint world: one object per entry of JOINT, anywhere on the ground.

    Objects of one colour keep the generator's minimum distance and margin among themselves; a red and a blue object may sit at any
    gap of 0 or more but their footprints never overlap; the generator's visibility check keeps every object at least
    min_pixels_per_object (200) visible pixels at 320 by 240. With MASK_SIZE, the objects' visible masks are written as well.
    """
    with open(args.properties_json, "r") as f:
        properties = json.load(f)
    colours = {name: [float(c) / 255.0 for c in rgb] + [1.0] for name, rgb in properties["colors"].items()}
    materials = [(v, k) for k, v in properties["materials"].items()]
    shapes = [(v, k) for k, v in properties["shapes"].items()]
    sizes = list(properties["sizes"].items())
    bpy.context.scene.update()
    scene_struct["asked_colours"] = list(JOINT)
    directions = [scene_struct["directions"][d] for d in ("left", "right", "front", "behind")]

    for attempt in range(MAX_TRIES):
        placed, objects, blender_objects = [], [], []
        fits = True
        for colour_name in JOINT:
            size_name, r = random.choice(sizes)
            obj_name, obj_name_out = random.choice(shapes)
            if obj_name == "Cube":
                r /= math.sqrt(2)
            theta = 360.0 * random.random()
            for _ in range(args.max_retries):
                x, y = random.uniform(-3, 3), random.uniform(-3, 3)
                good = True
                for other_colour, other_shape, xx, yy, rr, tt in placed:
                    if other_colour == colour_name:
                        dx, dy = x - xx, y - yy
                        if math.sqrt(dx * dx + dy * dy) - r - rr < args.min_dist:
                            good = False
                            break
                        if any(0 < dx * d[0] + dy * d[1] < args.margin for d in directions):
                            good = False
                            break
                    elif footprints_overlap((obj_name, x, y, r, theta), (other_shape, xx, yy, rr, tt)):
                        good = False
                        break
                if good:
                    break
            else:
                fits = False
                break
            utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
            obj = bpy.context.object
            blender_objects.append(obj)
            placed.append((colour_name, obj_name, x, y, r, theta))
            mat_name, mat_name_out = random.choice(materials)
            utils.add_material(mat_name, Color=colours[colour_name])
            objects.append({"shape": obj_name_out, "size": size_name, "material": mat_name_out,
                            "3d_coords": tuple(obj.location), "rotation": theta,
                            "pixel_coords": utils.get_camera_coords(camera, obj.location), "color": colour_name})
        if fits and (not blender_objects or generator.check_visibility(blender_objects, args.min_pixels_per_object)):
            if MASK_SIZE:
                write_masks(blender_objects, MASK_PATH[0], args)
                scene_struct["mask_file"] = os.path.basename(MASK_PATH[0])
            return objects, blender_objects
        for obj in blender_objects:
            utils.delete_object(obj)
    raise RuntimeError("could not place the objects in %d tries" % MAX_TRIES)


class _KnownColours(object):
    """Stands in for the random module inside the generator's render_shadeless, so object i is drawn in the known flat colour
    (i + 1) / (n + 1) on red, 1 on green and 0.5 on blue; green 1 separates objects from the background."""

    def __init__(self, n):
        self.values = []
        for i in range(n):
            self.values += [(i + 1.0) / (n + 1.0), 1.0, 0.5]

    def random(self):
        return self.values.pop(0)


def write_masks(blender_objects, path, args):
    """Each object's visible mask at MASK_SIZE by MASK_SIZE, from one shadeless render with a known flat colour per object.

    The render (no antialiasing, lamps and ground hidden by the generator's render_shadeless) is read back with Blender's numpy, each
    pixel given to the object whose flat colour it carries (objects are green 1; their red values, sorted, follow object order), and
    each object's mask box-averaged to MASK_SIZE: rows are repeated 4 times (240 to 960) so both sides divide evenly, then each
    15-row by 5-column block is averaged. Saved as <path>: an (objects, MASK_SIZE, MASK_SIZE) uint8 array of covered share times 255.
    """
    import numpy
    flat = path[:-len(".npz")] + "-flat.png"
    real_random = generator.random
    generator.random = _KnownColours(len(blender_objects))
    try:
        generator.render_shadeless(blender_objects, path=flat)
    finally:
        generator.random = real_random
    image = bpy.data.images.load(flat)
    w, h = image.size
    px = numpy.array(image.pixels[:]).reshape(h, w, 4)[::-1]       # Blender stores rows bottom up
    bpy.data.images.remove(image)
    is_object = px[..., 1] > 0.75
    reds = numpy.round(px[..., 0][is_object] * 255).astype(int)
    levels = sorted(set(reds.tolist()))
    if len(levels) != len(blender_objects):
        raise RuntimeError("mask render shows %d flat colours for %d objects" % (len(levels), len(blender_objects)))
    red_level = numpy.round(px[..., 0] * 255).astype(int)
    if h * 4 % MASK_SIZE or w % MASK_SIZE:
        raise RuntimeError("masks need 4 h and w divisible by %d, got %d by %d" % (MASK_SIZE, w, h))
    masks = []
    for level in levels:
        m = (is_object & (red_level == level)).astype(float).repeat(4, axis=0)
        bh, bw = (4 * h) // MASK_SIZE, w // MASK_SIZE
        masks.append(m.reshape(MASK_SIZE, bh, MASK_SIZE, bw).mean(axis=(1, 3)))
    numpy.savez_compressed(path, masks=numpy.round(numpy.stack(masks) * 255).astype(numpy.uint8))
    os.remove(flat)


def parse_targets(text):
    return [tuple(float(v) for v in item.split(",")) for item in text.split(";") if item.strip()]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--targets", default="")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--use-gpu", type=int, default=0, help="1 renders with Cycles on the GPU through CUDA")
    ap.add_argument("--colours", default="", help="Stage 2: one colour per object, such as red,red,blue, placed anywhere")
    ap.add_argument("--joint", default="", help="Paper 1: R,B red and blue objects together, touching allowed, never interpenetrating; "
                                                "'cycle' gives scene i (i mod 3) + 1 of each, sizes in turn")
    ap.add_argument("--masks", type=int, default=0, help="Paper 1: with --joint, write each object's visible mask at this size (64)")
    ours = ap.parse_args(argv)
    if ours.joint:
        if ours.targets or ours.colours:
            raise SystemExit("give --joint alone, not with --targets or --colours")
        if ours.seed_base < 8000000:
            raise SystemExit("joint renders start at seed base 8,000,000; earlier jobs used 1,000,000 to 7,330,249")
        return joint_main(ours)

    args = generator.parser.parse_args(["--use_gpu", str(ours.use_gpu), "--render_tile_size", str(ours.tile)])
    TARGETS[:] = parse_targets(ours.targets)
    COLOURS[:] = [c.strip() for c in ours.colours.split(",") if c.strip()]
    if TARGETS and COLOURS:
        raise SystemExit("give --targets or --colours, not both")
    generator.add_random_objects = colour_objects if COLOURS else place_objects
    count_objects = len(COLOURS) if COLOURS else len(TARGETS)
    folder = os.path.join(ours.out, ours.name)
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "scenes"), exist_ok=True)

    begun = time.time()
    done = 0
    for index in range(ours.start, ours.start + ours.count):
        stem = "%s_%06d" % (ours.name, index)
        image_path = os.path.join(folder, "images", stem + ".png")
        scene_path = os.path.join(folder, "scenes", stem + ".json")
        if os.path.exists(image_path) and os.path.exists(scene_path):
            continue
        random.seed(ours.seed_base + index)
        generator.render_scene(args, num_objects=count_objects, output_index=index, output_split=ours.name,
                               output_image=image_path, output_scene=scene_path)
        done += 1
        print("M7RENDER %s %d done=%d seconds_per_image=%.2f" % (ours.name, index, done, (time.time() - begun) / done), flush=True)
    print("M7RENDER finished %s rendered=%d" % (ours.name, done), flush=True)


def joint_counts(spec, index):
    """(red, blue) for scene index: 'cycle' is (i mod 3) + 1 of each, else the fixed 'R,B'."""
    if spec == "cycle":
        k = index % 3 + 1
        return k, k
    red, blue = (int(v) for v in spec.split(","))
    return red, blue


def joint_main(ours):
    """Paper 1's joint renders: images, scene files and, with --masks, masks/<stem>.npz, one scene per index, resumable."""
    global MASK_SIZE
    MASK_SIZE = ours.masks
    args = generator.parser.parse_args(["--use_gpu", str(ours.use_gpu), "--render_tile_size", str(ours.tile)])
    generator.add_random_objects = joint_objects
    folder = os.path.join(ours.out, ours.name)
    for sub in ("images", "scenes", "masks"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    # The joint renders' settings beside the experts' (m7_render.sbatch rendered them with --use-gpu 0 and --tile 32), at the run's top
    # level so the job copies it: the generator's every setting, camera and light jitter included, from its own parser.
    experts = generator.parser.parse_args(["--use_gpu", "0", "--render_tile_size", "32"])
    with open(os.path.join(os.path.dirname(os.path.abspath(ours.out)), "settings-%s.txt" % ours.name), "w") as f:
        f.write("setting | joint (%s) | experts (m7_render.sbatch) | same\n" % ours.name)
        for key in sorted(vars(args)):
            a, b = getattr(args, key), getattr(experts, key)
            f.write("%s | %s | %s | %s\n" % (key, a, b, "yes" if a == b else "NO"))
        f.write("blender | %s | the same binary in both job scripts | yes\n" % bpy.app.version_string)
    begun = time.time()
    done = 0
    for index in range(ours.start, ours.start + ours.count):
        stem = "%s_%06d" % (ours.name, index)
        image_path = os.path.join(folder, "images", stem + ".png")
        scene_path = os.path.join(folder, "scenes", stem + ".json")
        mask_path = os.path.join(folder, "masks", stem + ".npz")
        if os.path.exists(image_path) and os.path.exists(scene_path) and (not MASK_SIZE or os.path.exists(mask_path)):
            continue
        red, blue = joint_counts(ours.joint, index)
        JOINT[:] = ["red"] * red + ["blue"] * blue
        MASK_PATH[0] = mask_path
        random.seed(ours.seed_base + index)
        generator.render_scene(args, num_objects=len(JOINT), output_index=index, output_split=ours.name,
                               output_image=image_path, output_scene=scene_path)
        done += 1
        print("P1RENDER %s %d red=%d blue=%d done=%d seconds_per_image=%.2f" % (ours.name, index, red, blue, done,
                                                                             (time.time() - begun) / done), flush=True)
    print("P1RENDER finished %s rendered=%d" % (ours.name, done), flush=True)


if __name__ == "__main__":
    main()
