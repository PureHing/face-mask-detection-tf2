# -*- coding: utf-8 -*-
# @Time : 2020/3/20
# @File : check_dataset.py
# @Software: PyCharm
import cv2, time, os
import numpy as np
import sys
import os
rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(rootPath)

from components import config
from components.utils import  draw_anchor, decode_tf
from components.prior_box import priors_box
from dataset.tf_dataset_preprocess import load_tfrecord_dataset




def draw(img, ann, img_height, img_width, class_list):
    """draw bboxes and labels"""
    # bbox

    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence

    class_name = class_list[int(ann[-1])]
    cv2.putText(img, '{}'.format(class_name), (int(ann[0] * img_width), int(ann[1] * img_height) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))


def data_visulization():
    dataset = load_tfrecord_dataset(tfrecord_name=tfrecord_name, batch_size=batch_size, img_dim=cfg['input_size'],
                                          using_flip=True, using_distort=True, using_encoding=using_encoding,
                                          using_normalizing=using_normalizing, priors=priors, match_thresh=match_thresh,
                                          variances=variances, shuffle=False,repeat=False)

    start_time = time.time()
    #check dataset
    for idx, (inputs, labels) in enumerate(dataset.take(num_samples)):
        print("{} inputs:".format(idx), inputs.shape, "labels:", labels.shape)
        if not visualization:
            continue
        #img:-0.5~0.5 >> 0~255
        img = np.clip((inputs.numpy()[0]+0.5) * 255.0, 0, 255).astype(np.uint8)

        if not using_encoding:
            # labels includes loc,conf

            targets = labels.numpy()[0]

            for target in targets:

                draw(img, target, cfg['input_size'][0], cfg['input_size'][1], class_list)
        else:
            # labels includes loc, conf.
            targets = decode_tf(labels[0], priors, variances=variances).numpy()

            for prior_index in range(len(targets)):

                if targets[prior_index][4] > 0:
                    # print(f"class id {targets[prior_index][4]}")
                    draw(img, targets[prior_index], cfg['input_size'][0], cfg['input_size'][1], class_list)
                    draw_anchor(img, priors[prior_index], cfg['input_size'][0], cfg['input_size'][1])

        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) == ord('q') :
            exit()

    print("data fps: {:.2f}".format(num_samples / (time.time() - start_time)))


if __name__ == '__main__':

    #for test dataset
    cfg = config.cfg
    class_list = cfg['labels_list']
    print(f"class:{class_list}")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

    batch_size = 1
    priors, num_cell = priors_box(cfg)
    visualization = True  # False for time cost estimattion
    using_encoding = True  # batch size should be 1 when False
    using_normalizing = True  # image:-0.5~0.5
    variances = [0.1, 0.2]
    match_thresh = 0.45
    ignore_thresh = 0.3

    num_samples = cfg['dataset_len']
    tfrecord_name = rootPath+'/dataset/train_mask.tfrecord'

    # num_samples = cfg['val_len']
    # tfrecord_name = rootPath+'/dataset/trainval_mask.tfrecord'

    data_visulization()
    exit()
