# -*- coding: utf-8 -*-
# @Time : 2020/3/21 
# @File : inference.py
# @Software: PyCharm
import cv2
import os
import time

import numpy as np
import tensorflow as tf
from absl import flags, app
from absl.flags import FLAGS

from components import config
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms, pad_input_image, recover_pad_output, show_image
from network.network import SlimModel  # defined by tf.keras

flags.DEFINE_string('model_path', 'checkpoints/', 'config file path')
flags.DEFINE_string('img_path', 'assets/1_Handshaking_Handshaking_1_71.jpg', 'path to input image')
flags.DEFINE_boolean('camera', True, 'get image source from webcam or not')


def parse_predict(predictions, priors, cfg):
    label_classes = cfg['labels_list']

    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg['variances'])

    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]

        score_idx = cls_scores > cfg['score_threshold']

        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, cfg['nms_threshold'], cfg['max_number_keep'])

        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)

        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


def main(_):
    global model
    cfg = config.cfg
    min_sizes = cfg['min_sizes']
    num_cell = [len(min_sizes[k]) for k in range(len(cfg['steps']))]

    try:
        model = SlimModel(cfg=cfg, num_cell=num_cell, training=False)

        paths = [os.path.join(FLAGS.model_path, path)
                 for path in os.listdir(FLAGS.model_path)]
        latest = sorted(paths, key=os.path.getmtime)[-1]
        model.load_weights(latest)
        print(f"model path : {latest}")
        model.save('final.h5') #if want to convert to tflite by model.save,it should be set input image size.
        # model.summary()
    except AttributeError as e:
        print('Please make sure there is at least one weights at {}'.format(FLAGS.model_path))

    if not FLAGS.camera:
        if not os.path.exists(FLAGS.img_path):
            print(f"Cannot find image path from {FLAGS.img_path}")
            exit()
        print("[*] Predict {} image.. ".format(FLAGS.img_path))
        img_raw = cv2.imread(FLAGS.img_path)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
        img = img / 255.0 - 0.5

        priors, _ = priors_box(cfg, image_sizes=(img.shape[0], img.shape[1]))
        priors = tf.cast(priors, tf.float32)

        predictions = model.predict(img[np.newaxis, ...])

        boxes, classes, scores = parse_predict(predictions, priors, cfg)

        print(f"scores:{scores}")
        # recover padding effect
        boxes = recover_pad_output(boxes, pad_params)

        # draw and save results
        save_img_path = os.path.join('assets/out_' + os.path.basename(FLAGS.img_path))

        for prior_index in range(len(boxes)):
            show_image(img_raw, boxes, classes, scores, img_height_raw, img_width_raw, prior_index, cfg['labels_list'])

        cv2.imwrite(save_img_path, img_raw)
        cv2.imshow('results', img_raw)
        if cv2.waitKey(0) == ord('q'):
            exit(0)

    else:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        priors, _ = priors_box(cfg, image_sizes=(240, 320))
        priors = tf.cast(priors, tf.float32)
        start = time.time()
        while True:
            _, frame = capture.read()
            if frame is None:
                print('No camera found')

            h, w, _ = frame.shape
            img = np.float32(frame.copy())

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img / 255.0 - 0.5

            predictions = model(img[np.newaxis, ...])
            boxes, classes, scores = parse_predict(predictions, priors, cfg)

            for prior_index in range(len(classes)):
                show_image(frame, boxes, classes, scores, h, w, prior_index, cfg['labels_list'])
            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start))
            start = time.time()
            cv2.putText(frame, fps_str, (25, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        app.run(main)
    except Exception as e:
        print(e)
        exit()
