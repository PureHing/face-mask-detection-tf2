# -*- coding: utf-8 -*-
# @Time : 2020/3/29 
# @File : detect.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import os, cv2, tqdm, sys
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.keras.models import load_model
import xml.etree.ElementTree as ET

rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(rootPath)

from components import config


from network.network import SlimModel
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms

flags.DEFINE_string('model_path', 'checkpoints/', 'VOC format dataset')
flags.DEFINE_string('dataset_path', 'Maskdata', 'VOC format dataset')
flags.DEFINE_enum('split', 'val', ['val', 'trainval'], 'val or test dataset')
flags.DEFINE_list('image_size',[240,320],'single scale for model test')

def parse_predict(predictions, priors, cfg):
    label_classes = cfg['labels_list']

    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg['variances'])
    ##classifications shape :(num_priors,num_classes)

    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]

        score_idx = cls_scores > 0.02#cfg['score_threshold']

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


def parse_annot_gt(annot_file):
    """Parse Pascal VOC annotations."""
    tree = ET.parse(annot_file)
    root = tree.getroot()


    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if not difficult:
            difficult = '0'
        else:
            difficult = difficult.text
        cls = obj.find('name').text

        xmlbox = obj.find('bndbox')
        bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text),int(xmlbox.find('ymax').text))
        list_file.write(cls + " ".join([str(a) for a in bbox]))







def main(_):
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    detect_reslut_dir = 'mAP/detection-results/'
    if not os.path.exists(detect_reslut_dir):
        os.makedirs(detect_reslut_dir)

    for file in os.listdir(detect_reslut_dir):
        path_file = os.path.join(detect_reslut_dir + file)
        if os.path.isfile(path_file):
            os.remove(path_file)

    ground_thuth_dir = 'mAP/ground-truth/'
    if not os.path.exists(ground_thuth_dir):
        os.makedirs(ground_thuth_dir)

    for file in os.listdir(ground_thuth_dir):
        path_file = os.path.join(ground_thuth_dir+file)
        if os.path.isfile(path_file):
            os.remove(path_file)

    logging.info('Reading configuration...')
    cfg = config.cfg
    class_list = cfg['labels_list']

    image_size = tuple(FLAGS.image_size)

    logging.info("Class dictionary loaded: %s", class_list)

    priors, num_cell = priors_box(cfg, image_size)
    priors = tf.cast(priors, tf.float32)

    try:
        model = load_model(FLAGS.model_path)
    except:
        model = SlimModel(cfg=cfg, num_cell=num_cell, training=False)
        paths = [os.path.join(FLAGS.model_path, path)
                 for path in os.listdir(FLAGS.model_path)]
        latest = sorted(paths, key=os.path.getmtime)[-1]
        model.load_weights(latest)
        print(f"model path : {latest}")

    img_list = open(
        os.path.join(FLAGS.dataset_path, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
    logging.info("Image list loaded: %d", len(img_list))

    for image in tqdm.tqdm(img_list):

        image_file = os.path.join(FLAGS.dataset_path, 'JPEGImages', '%s.jpg' % image)
        annot_file = os.path.join(FLAGS.dataset_path, 'Annotations', '%s.xml' % image)

        # detect image
        img_raw = cv2.imread(image_file)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())
        img = cv2.resize(img, (image_size[1], image_size[0])) # cv2.resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0 - 0.5) / 1.0
        predictions = model.predict(img[np.newaxis, ...])

        boxes, classes, scores = parse_predict(predictions, priors, cfg)
        with open( detect_reslut_dir+ f'{image}.txt', "a") as new_f:
            for prior_index in range(len(boxes)):
                x1, y1, x2, y2 = (boxes[prior_index][0] * img_width_raw), (boxes[prior_index][1] * img_height_raw), \
                                 (boxes[prior_index][2] * img_width_raw), (boxes[prior_index][3] * img_height_raw)

                top = max(0, np.floor( y1+ 0.5).astype('int32'))
                left = max(0, np.floor( x1+ 0.5).astype('int32'))
                bottom = min(img_width_raw, np.floor(y2 + 0.5).astype('int32'))
                right = min(img_height_raw, np.floor( x2+ 0.5).astype('int32'))

                class_name = class_list[classes[prior_index]]
                score = "{:.2f}".format(scores[prior_index])
                label = '{} {}'.format(class_name, score)
                new_f.write("%s %s %s %s %s\n" % (label, left,top, right,bottom))


        # ground truth
        with open(ground_thuth_dir + f'{image}.txt', 'a') as gt_f:
            tree = ET.parse(annot_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                difficult = obj.find('difficult')
                if not difficult:
                    difficult = '0'
                else:
                    difficult = difficult.text
                cls = obj.find('name').text

                xmlbox = obj.find('bndbox')
                bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                        int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
                gt_f.write(cls +' '+ " ".join([str(a) for a in bbox])+'\n')



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        app.run(main)
    except Exception as e:
        print(e)
        exit()
