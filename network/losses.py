# -*- coding: utf-8 -*-
# @Time : 2020/3/19
# @File : losses.py
# @Software: PyCharm

import tensorflow as tf


def _smooth_l1_loss(y_true, y_pred):
    t = tf.abs(y_pred - y_true)
    return tf.where(t < 1, 0.5 * t ** 2, t - 0.5)


def sigmoid_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    pass


def hard_negative_mining(loss, class_truth, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B, num_default)
        class_truth: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        class_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # class_truth: B x N
    pos_idx = class_truth > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


def MultiBoxLoss(num_class=3, neg_pos_ratio=3.0):
    def multi_loss(y_true, y_pred):
        """ Compute losses for SSD
               regression loss: smooth L1
               classification loss: cross entropy
           Args:
               y_true: [B,N,5]
               y_pred: [B,N,num_class]
               class_pred: outputs of classification heads (B,N, num_classes)
               loc_pred: outputs of regression heads (B,N, 4)
               class_truth: classification targets (B,N)
               loc_truth: regression targets (B,N, 4)
           Returns:
               class_loss: classification loss
               loc_loss: regression loss
       """
        num_batch = tf.shape(y_true)[0]
        num_prior = tf.shape(y_true)[1]
        loc_pred, class_pred = y_pred[..., :4], y_pred[..., 4:]
        loc_truth, class_truth = y_true[..., :4], tf.squeeze(y_true[..., 4:])
        # print(f"loc_pred{y_pred}")
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # compute classification losses without reduction
        temp_loss = cross_entropy(class_truth, class_pred)
        # 2. hard negative mining
        pos_idx, neg_idx = hard_negative_mining(temp_loss, class_truth, neg_pos_ratio)

        # classification loss will consist of positive and negative examples
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')

        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        loss_class = cross_entropy(
            class_truth[tf.math.logical_or(pos_idx, neg_idx)],
            class_pred[tf.math.logical_or(pos_idx, neg_idx)])

        # localization loss only consist of positive examples (smooth L1)
        loss_loc = smooth_l1_loss(loc_truth[pos_idx],loc_pred[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        loss_class = loss_class / num_pos
        loss_loc = loss_loc / num_pos
        return loss_loc, loss_class

    return multi_loss
