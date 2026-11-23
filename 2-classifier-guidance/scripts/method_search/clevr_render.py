"""Render CLEVR scenes with each object placed at a chosen image position (step M7).

Runs inside Blender 2.78c, whose Python is 3.5. It calls the released generator's render_scene unchanged and replaces only the step
that picks where objects go. Shapes, sizes, colours, materials, camera jitter and light jitter stay the generator's defaults.

A target (x, y) is a fraction of the image width and height, measured from the top left. Each object's target gets a uniform offset of up
to OFFSET, and the object goes on the ground point that projects there under this image's jittered camera. Each image has its own seed,
and an image whose scene file exists is skipped, so a job can resume.

Usage:
  CLEVR_GEN=<image_generation folder> blender --background --threads N --python clevr_render.py -- \
      --name p1 --targets "0.30,0.30" --seed-base 1000000 --start 0 --count 8 --out <dir>
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
    ours = ap.parse_args(argv)

    args = generator.parser.parse_args(["--use_gpu", "0", "--render_tile_size", str(ours.tile)])
    TARGETS[:] = parse_targets(ours.targets)
    generator.add_random_objects = place_objects
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
        generator.render_scene(args, num_objects=len(TARGETS), output_index=index, output_split=ours.name,
                               output_image=image_path, output_scene=scene_path)
        done += 1
        print("M7RENDER %s %d done=%d seconds_per_image=%.2f" % (ours.name, index, done, (time.time() - begun) / done), flush=True)
    print("M7RENDER finished %s rendered=%d" % (ours.name, done), flush=True)


if __name__ == "__main__":
    main()
