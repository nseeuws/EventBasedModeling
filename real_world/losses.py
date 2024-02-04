from typing import Callable
import tensorflow as tf 


def stable_softplus_tf(x):
    return tf.math.softplus(-tf.math.abs(x)) + tf.math.maximum(x, 0)

@tf.function
def regression_loss_tf_iou(target_size: tf.Tensor, pred_size: tf.Tensor) -> tf.Tensor:
    """Compute a duration regression loss, based on IoU

    Args:
        target_size (tf.Tensor): Duration targets
        pred_size (tf.Tensor): Duration predictions

    Returns:
        tf.Tensor: Scalar regression loss
    """
    mask = target_size!=0
    target_ = tf.boolean_mask(target_size, mask=mask)
    pred_ = tf.boolean_mask(pred_size, mask=mask)

    iou = tf.math.minimum(x=target_, y=pred_) / tf.math.maximum(x=target_, y=pred_)
    iou_loss = tf.reduce_sum(1. - iou)
    return iou_loss


@tf.function
def focal_loss_tf(loc_map: tf.Tensor, pred_loc: tf.Tensor, pred_logit: tf.Tensor,
        alpha: float = 2., beta: float = 4., a_t: float = 0.1) -> tf.Tensor:
    """Compute centerpoint prediction focal loss

    Args:
        loc_map (tf.Tensor): Kernel-smoothed ground truth
        pred_loc (tf.Tensor): Center point prediction probabilities
        pred_logit (tf.Tensor): Logits of the center point prediction.
        Used to compute a numerically stable focal loss
        alpha (float, optional): Focal loss exponent. Defaults to 2.
        beta (float, optional): Background smoothing exponent. Defaults to 4.
        a_t (float, optional): Weight of the background terms. Foreground
        is weighted as `1 - a_t`. Defaults to 0.1

    Returns:
        [type]: [description]
    """
    one = tf.constant(1., dtype=tf.float32)
    background_term = tf.math.multiply(tf.math.pow(one - loc_map, beta), tf.math.pow(pred_loc, alpha))
    background_term = tf.math.multiply(background_term, -stable_softplus_tf(pred_logit))
    background_term = a_t*tf.reduce_sum(background_term)

    #mask = loc_map==1
    mask = tf.where(loc_map==1, tf.constant(True), tf.constant(False))
    loc_ = tf.boolean_mask(pred_loc, mask=mask)
    logit_ = tf.boolean_mask(pred_logit, mask=mask)
    center_term = tf.math.multiply(tf.math.pow(1. - loc_, alpha), logit_ - stable_softplus_tf(logit_))
    center_term = (1.-a_t)*tf.reduce_sum(center_term)
    
    return -(background_term + center_term)


def get_focal_loss(alpha: float = 2., beta: float = 4., a_t=0.1) -> Callable:
    """Create a focal loss "object" with the given parameters

    Args:
        alpha (float, optional): Focal loss exponent. Defaults to 2.
        beta (float, optional): Background smoothing exponent. Defaults to 4.
        a_t (float, optional): Weight of the background terms. 
        Foreground is weighted as `1-a_t`. Defaults to 0.1.

    Returns:
        function: Function to compute the focal loss
    """
    @tf.function
    def focal_loss(loc_map: tf.Tensor, pred_loc: tf.Tensor, pred_logit: tf.Tensor) -> tf.Tensor:
        loss = focal_loss_tf(
            loc_map=loc_map, pred_loc=pred_loc, pred_logit=pred_logit,
            alpha=alpha, beta=beta, a_t=a_t
        )
        return loss
    return focal_loss