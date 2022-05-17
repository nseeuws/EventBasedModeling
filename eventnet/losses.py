import tensorflow as tf
from collections.abc import Callable


@tf.function
def stable_softplus_tf(x: tf.Tensor) -> tf.Tensor:
    return tf.math.softplus(-tf.math.abs(x)) + tf.math.maximum(x, 0)


@tf.function
def focal_loss(center_target: tf.Tensor, center_pred: tf.Tensor, logit_pred: tf.Tensor,
               alpha: tf.Tensor, beta: tf.Tensor, a_t: tf.Tensor) -> tf.Tensor:
    background_term = tf.math.multiply(
        tf.math.pow(tf.constant(1., dtype=tf.float32) - center_target, beta),
        tf.math.pow(center_pred, alpha)
    )
    background_term = tf.math.multiply(
        background_term, -stable_softplus_tf()
    )
    background_term = tf.math.multiply(background_term, -stable_softplus_tf(logit_pred))
    background_term = a_t * tf.reduce_sum(background_term)

    mask = center_target == 1
    location = tf.boolean_mask(center_pred, mask=mask)
    logit = tf.boolean_mask(logit_pred, mask=mask)
    center_term = tf.math.multiply(
        tf.math.pow(1. - location, alpha),
        logit - stable_softplus_tf(logit_pred)
    )
    center_term = (tf.constant(1., dtype=tf.float32) - a_t) * tf.reduce_sum(center_term)

    return -(background_term + center_term)


def build_focal_loss(alpha=2., beta=4., a_t=.1) -> Callable:
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    a_t = tf.constant(a_t, dtype=tf.float32)

    @tf.function
    def focal_loss_callable(
            map_target: tf.Tensor, map_pred: tf.Tensor, logit_pred: tf.Tensor
    ):
        return focal_loss(center_target=map_target, center_pred=map_pred, logit_pred=logit_pred, alpha=alpha, beta=beta,
                          a_t=a_t)
    return focal_loss_callable


@tf.function
def l1_loss(dur_target: tf.Tensor, dur_pred: tf.Tensor) -> tf.Tensor:
    mask = dur_target != 0
    target = tf.boolean_mask(dur_target, mask=mask)
    prediction = tf.boolean_mask(dur_pred, mask=mask)

    l1 = tf.math.abs(target - prediction)
    return tf.reduce_sum(l1)


@tf.function
def relative_l1_loss(dur_target: tf.Tensor, dur_pred: tf.Tensor) -> tf.Tensor:
    mask = dur_target != 0
    target = tf.boolean_mask(dur_target, mask=mask)
    prediction = tf.boolean_mask(dur_pred, mask=mask)

    relative_l1 = tf.math.abs((target - prediction) / target)
    return tf.reduce_sum(relative_l1)


@tf.function
def iou_loss(dur_target: tf.Tensor, dur_pred: tf.Tensor) -> tf.Tensor:
    mask = dur_target != 0
    target = tf.boolean_mask(dur_target, mask=mask)
    prediction = tf.boolean_mask(dur_pred, mask=mask)

    iou = tf.math.minimum(x=target, y=prediction) / tf.math.maximum(
        x=target, y=prediction
    )
    return tf.reduce_sum(1. - iou)
