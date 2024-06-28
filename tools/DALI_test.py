from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import os.path

test_data_root = os.environ["DALI_EXTRA_PATH"]
file_root = os.path.join(test_data_root, "db", "coco", "images")
annotations_file = os.path.join(test_data_root, "db", "coco", "instances.json")
batch_size = 16

pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
with pipe:
    jpegs, bboxes, labels, polygons, vertices = fn.readers.coco(
        file_root=file_root, annotations_file=annotations_file, polygon_masks=True, ratio=True
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    pipe.set_outputs(images, bboxes, labels, polygons, vertices)

pipe.build()
pipe_out = pipe.run()

images_cpu = pipe_out[0].as_cpu()
bboxes_cpu = pipe_out[1]
labels_cpu = pipe_out[2]
polygons_cpu = pipe_out[3]
vertices_cpu = pipe_out[4]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

random.seed(1231243)


def plot_sample(img_index, ax):
    img = images_cpu.at(img_index)

    H = img.shape[0]
    W = img.shape[1]

    ax.imshow(img)
    bboxes = bboxes_cpu.at(img_index)
    labels = labels_cpu.at(img_index)
    polygons = polygons_cpu.at(img_index)
    vertices = vertices_cpu.at(img_index)
    categories_set = set()
    for label in labels:
        categories_set.add(label)

    category_id_to_color = dict(
        [
            (cat_id, [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
            for cat_id in categories_set
        ]
    )

    for bbox, label in zip(bboxes, labels):
        rect = patches.Rectangle(
            (bbox[0] * W, bbox[1] * H),
            bbox[2] * W,
            bbox[3] * H,
            linewidth=1,
            edgecolor=category_id_to_color[label],
            facecolor="none",
        )
        ax.add_patch(rect)

    for polygon in polygons:
        mask_idx, start_vertex, end_vertex = polygon
        polygon_vertices = vertices[start_vertex:end_vertex]
        polygon_vertices = polygon_vertices * [W, H]
        poly = patches.Polygon(
            polygon_vertices, True, facecolor=category_id_to_color[label], alpha=0.7
        )
        ax.add_patch(
            poly,
        )


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
fig.tight_layout()
plot_sample(2, ax[0, 0])
plot_sample(1, ax[0, 1])
plot_sample(4, ax[1, 0])
plot_sample(8, ax[1, 1])
plt.show()