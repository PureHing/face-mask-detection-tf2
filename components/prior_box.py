# -*- coding: utf-8 -*-


import math,sys
import numpy as np
from itertools import product as product


def priors_box(cfg,image_sizes=None):
    """prior box"""
    if image_sizes is None:
        image_sizes = cfg['input_size']
    min_sizes=cfg["min_sizes"]
    steps=cfg["steps"]
    clip=cfg["clip"]

    if isinstance(image_sizes, int):
        image_sizes = (image_sizes, image_sizes)
    elif isinstance(image_sizes, tuple):
        image_sizes = image_sizes
    else:
        raise Exception('Type error of input image size format,tuple or int. ')

    for m in range(4):
        if (steps[m] != pow(2, (m + 3))):
            print("steps must be [8,16,32,64]")
            sys.exit()

    assert len(min_sizes) == len(steps), "anchors number didn't match the feature map layer."

    feature_maps = [
        [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
        for step in steps]

    anchors = []
    num_box_fm_cell=[]
    for k, f in enumerate(feature_maps):
        num_box_fm_cell.append(len(min_sizes[k]))
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                if isinstance(min_size, int):
                    min_size = (min_size, min_size)
                elif isinstance(min_size, tuple):
                    min_size=min_size
                else:
                    raise Exception('Type error of min_sizes elements format,tuple or int. ')
                s_kx = min_size[1] / image_sizes[1]
                s_ky = min_size[0] / image_sizes[0]
                cx = (j + 0.5) * steps[k] / image_sizes[1]
                cy = (i + 0.5) * steps[k] / image_sizes[0]
                anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4])
    # print("prios:",output.shape,len(output))
    # print("num box for fm cell:",num_box_fm_cell)
    if clip:
        output = np.clip(output, 0, 1)
    return output,num_box_fm_cell



if __name__ == '__main__':
    import config
    priors_box(config.cfg)