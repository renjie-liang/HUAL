import re
import numpy as np
import tensorflow as tf


def gumbel_sample(shape):
    g_sample = tf.random.uniform(shape, minval=0, maxval=1)
    noise = -tf.math.log(-tf.math.log(g_sample + 1e-20) + 1e-20)
    return noise


def gumbel_softmax(logits, tau, hard=False):
    # from https://github.com/vithursant/VAE-Gumbel-Softmax/blob/master/vae_gumbel_softmax.py
    # generate gumbel noise
    g_sample = tf.random.uniform(tf.shape(input=logits), minval=0, maxval=1)
    noise = -tf.math.log(-tf.math.log(g_sample + 1e-20) + 1e-20)
    # compute gumbel softmax
    y = tf.nn.softmax((logits + noise) / tau)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(input_tensor=y, axis=1, keepdims=True)), dtype=y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def gumbel_sigmoid(logits, tau, hard=False):
    # implementation of How Does Selective Mechanism Improve Self-Attention Networks?
    # https://arxiv.org/pdf/2005.00979.pdf
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    # https://timvieira.github.io/blog/post/2016/07/04/fast-sigmoid-sampling/
    # https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py
    g_sample1 = tf.random.uniform(tf.shape(input=logits), minval=0, maxval=1)
    g_sample2 = tf.random.uniform(tf.shape(input=logits), minval=0, maxval=1)
    noise = -tf.math.log(tf.math.log(g_sample2 + 1e-20) / tf.math.log(g_sample1 + 1e-20) + 1e-20)
    y = tf.sigmoid((logits + noise) / tau)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(input_tensor=y, axis=1, keepdims=True)), dtype=y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def label_smoothing(labels, mask, epsilon=0.1):
    mask = tf.cast(mask, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)  # (batch_size, label_size)
    seq_len = tf.reduce_sum(input_tensor=mask, axis=1, keepdims=False)  # (batch_size)
    smooth_labels = (1.0 - epsilon) * labels + tf.expand_dims(epsilon / seq_len, axis=1)
    smooth_labels = smooth_labels * mask  # keep the padding position as zero
    return smooth_labels


def count_params(scope=None):
    if scope is None:
        return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    else:
        return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables(scope)]))


def get_shape_list(tensor):
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def transpose_for_scores(input_tensor, batch_size, seq_length, num_heads, head_size):
    output_tensor = tf.reshape(input_tensor, shape=[batch_size, seq_length, num_heads, head_size])
    output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
    return output_tensor


def create_attention_mask(from_mask, to_mask, broadcast_ones=False):
    batch_size, from_seq_len = get_shape_list(from_mask)
    _, to_seq_len = get_shape_list(to_mask)
    to_mask = tf.cast(tf.expand_dims(to_mask, axis=1), dtype=tf.float32)
    if broadcast_ones:
        mask = tf.ones(shape=[batch_size, from_seq_len, 1], dtype=tf.float32)
    else:
        mask = tf.cast(tf.expand_dims(from_mask, axis=2), dtype=tf.float32)
    mask = tf.matmul(mask, to_mask)  # (batch_size, from_seq_len, to_seq_len)
    return mask


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, dtype=inputs.dtype)
    return inputs * mask + mask_value * (1.0 - mask)


def trilinear_attention(args, maxlen1, maxlen2, drop_rate=0.0, reuse=None, name='efficient_trilinear'):
    assert len(args) == 2, 'just use for computing attention with two input'
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError('`args` must be 3 dims (batch_size, len, dimension)')
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError('the last dimension of `args` must equal')
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    drop_args = [tf.nn.dropout(arg, rate=drop_rate) for arg in args]
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        weights4arg0 = tf.compat.v1.get_variable('linear_kernel4arg0', [arg_size, 1], dtype=dtype)
        weights4arg1 = tf.compat.v1.get_variable('linear_kernel4arg1', [arg_size, 1], dtype=dtype)
        weights4mlu = tf.compat.v1.get_variable('linear_kernel4mul', [1, 1, arg_size], dtype=dtype)
        # compute results
        weights4arg0 = tf.tile(tf.expand_dims(weights4arg0, axis=0), multiples=[tf.shape(input=args[0])[0], 1, 1])
        subres0 = tf.tile(tf.matmul(drop_args[0], weights4arg0), [1, 1, maxlen2])
        weights4arg1 = tf.tile(tf.expand_dims(weights4arg1, axis=0), multiples=[tf.shape(input=args[1])[0], 1, 1])
        subres1 = tf.tile(tf.transpose(a=tf.matmul(drop_args[1], weights4arg1), perm=(0, 2, 1)), [1, maxlen1, 1])
        subres2 = tf.matmul(drop_args[0] * weights4mlu, tf.transpose(a=drop_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        return res


def create_optimizer(loss, lr, clip_norm=1.0):
    """Creates an optimizer training op."""
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = AdamWeightDecayOptimizer(learning_rate=lr, weight_decay_rate=0.01, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-6, exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    tvars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(ys=loss, xs=tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    # Normally the global step update is done inside of `apply_gradients`. However, `AdamWeightDecayOptimizer` doesn't
    # do this. But if you use a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self, learning_rate, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-6,
                 exclude_from_weight_decay=None, name='AdamWeightDecayOptimizer'):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            m = tf.compat.v1.get_variable(name=param_name + '/adam_m',
                                shape=param.shape.as_list(),
                                dtype=tf.float32,
                                trainable=False,
                                initializer=tf.compat.v1.zeros_initializer())
            v = tf.compat.v1.get_variable(name=param_name + '/adam_v',
                                shape=param.shape.as_list(),
                                dtype=tf.float32,
                                trainable=False,
                                initializer=tf.compat.v1.zeros_initializer())
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _apply_dense(self, grad, var):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass
