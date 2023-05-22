import math
import tensorflow as tf
from models.ops import get_shape_list
from models.layers import conv1d, layer_norm, depthwise_separable_conv, transpose_for_scores, \
    create_attention_mask, dual_multihead_attention


def word_embs(word_ids, dim, vectors, drop_rate=0.0, finetune=False, reuse=False, name='word_embs'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        table = tf.Variable(vectors, name='word_table', dtype=tf.float32, trainable=finetune)
        unk = tf.compat.v1.get_variable(name='unk', shape=[1, dim], dtype=tf.float32, trainable=True)
        zero = tf.zeros(shape=[1, dim], dtype=tf.float32)
        word_table = tf.concat([zero, unk, table], axis=0)
        word_emb = tf.nn.embedding_lookup(params=word_table, ids=word_ids)
        word_emb = tf.nn.dropout(word_emb, rate=drop_rate)
        return word_emb


def char_embs(char_ids, char_size, dim, kernels, filters, drop_rate=0.0, activation=tf.nn.relu, padding='VALID',
              reuse=False, name='char_embs'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # char embeddings lookup
        table = tf.compat.v1.get_variable(name='char_table', shape=[char_size - 1, dim], dtype=tf.float32, trainable=True)
        zero = tf.zeros(shape=[1, dim], dtype=tf.float32)
        char_table = tf.concat([zero, table], axis=0)
        char_emb = tf.nn.embedding_lookup(params=char_table, ids=char_ids)
        char_emb = tf.nn.dropout(char_emb, rate=drop_rate)
        # char-level cnn
        outputs = []
        for i, (kernel, channel) in enumerate(zip(kernels, filters)):
            weight = tf.compat.v1.get_variable('filter_%d' % i, shape=[1, kernel, dim, channel], dtype=tf.float32)
            bias = tf.compat.v1.get_variable('bias_%d' % i, shape=[channel], dtype=tf.float32, initializer=tf.compat.v1.zeros_initializer())
            output = tf.nn.conv2d(input=char_emb, filters=weight, strides=[1, 1, 1, 1], padding=padding, name='conv_%d' % i)
            output = tf.nn.bias_add(output, bias=bias)
            output = tf.reduce_max(input_tensor=activation(output), axis=2)
            outputs.append(output)
        outputs = tf.concat(values=outputs, axis=-1)
        return outputs


def add_pos_embs(inputs, max_pos_len, reuse=None, name='position_emb'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        batch_size, seq_length, dim = get_shape_list(inputs)
        assert_op = tf.compat.v1.assert_less_equal(seq_length, max_pos_len)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.compat.v1.get_variable(name='position_embeddings', shape=[max_pos_len, dim],
                                                       dtype=tf.float32)
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])
            num_dims = len(inputs.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, dim])
            position_embeddings = tf.reshape(position_embeddings, shape=position_broadcast_shape)
            outputs = inputs + position_embeddings
        return outputs


def conv_block(inputs, kernel_size, dim, num_layers, drop_rate=0.0, activation=tf.nn.relu, reuse=None,
               name='conv_block'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        outputs = tf.expand_dims(inputs, axis=2)
        for layer_idx in range(num_layers):
            residual = outputs
            outputs = layer_norm(outputs, reuse=reuse, name='layer_norm_%d' % layer_idx)
            outputs = depthwise_separable_conv(outputs, kernel_size=(kernel_size, 1), dim=dim, use_bias=True,
                                               activation=activation, name='depthwise_conv_layers_%d' % layer_idx,
                                               reuse=reuse)
            outputs = tf.nn.dropout(outputs, rate=drop_rate) + residual
        return tf.squeeze(outputs, 2)


def dual_attn_block(from_tensor, to_tensor, dim, num_heads, from_mask, to_mask, drop_rate, use_bias=True,
                    activation=None, reuse=None, name='dual_attention_block'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # multihead attention layer
        outputs = layer_norm(from_tensor, reuse=reuse, name='layer_norm_1')
        to_tensor = layer_norm(to_tensor, reuse=reuse, name='layer_norm_t')
        outputs = dual_multihead_attention(from_tensor=outputs, to_tensor=to_tensor, dim=dim, num_heads=num_heads,
                                           from_mask=from_mask, to_mask=to_mask, drop_rate=drop_rate, reuse=reuse,
                                           name='dual_multihead_attention')
        outputs = conv1d(outputs, dim=dim, use_bias=use_bias, activation=activation, reuse=reuse, name='dense_1')
        residual = tf.nn.dropout(outputs, rate=drop_rate) + from_tensor
        # feed forward layer
        outputs = layer_norm(residual, reuse=reuse, name='layer_norm_2')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        outputs = conv1d(outputs, dim=dim, use_bias=use_bias, reuse=reuse, name='dense_2')
        outputs = tf.nn.dropout(outputs, rate=drop_rate) + residual
        return outputs


def top_self_attention(inputs, dim, num_heads, mask, drop_rate, reuse=None, name='top_self_attention'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        if dim % num_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the attention heads (%d)' % (dim, num_heads))
        batch_size, seq_len, _ = get_shape_list(inputs)
        head_size = dim // num_heads
        # projection
        query = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='query')
        key = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='key')
        value = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='value')
        # reshape & transpose: (batch_size, seq_len, dim) --> (batch_size, num_heads, seq_len, head_size)
        query = transpose_for_scores(query, batch_size, seq_len, num_heads, head_size)
        key = transpose_for_scores(key, batch_size, seq_len, num_heads, head_size)
        value = transpose_for_scores(value, batch_size, seq_len, num_heads, head_size)
        # create attention mask
        attention_mask = create_attention_mask(mask, mask, broadcast_ones=False)
        attention_mask = tf.expand_dims(attention_mask, axis=1)  # (batch_size, 1, seq_len, seq_len)
        # compute attention score
        attention_value = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_len, seq_len)
        attention_value = tf.multiply(attention_value, 1.0 / math.sqrt(float(head_size)))
        attention_value += (1.0 - attention_mask) * -1e30
        attention_score = tf.nn.softmax(attention_value)
        attention_score = tf.nn.dropout(attention_score, rate=drop_rate)
        # compute value
        value = tf.matmul(attention_score, value)  # (batch_size, num_heads, from_seq_len, head_size)
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])
        value = tf.reshape(value, shape=[batch_size, seq_len, num_heads * head_size])
        return value


def feature_encoder(inputs, dim, num_heads, max_pos_len, drop_rate, attn_drop, mask, reuse=None,
                    name='feature_encoder'):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        features = add_pos_embs(inputs, max_pos_len=max_pos_len, reuse=reuse, name='pos_emb')
        features = conv_block(features, kernel_size=7, dim=dim, num_layers=4, reuse=reuse, drop_rate=drop_rate,
                              name='conv_block')
        with tf.compat.v1.variable_scope('multihead_attention_block', reuse=reuse):
            # multihead attention layer
            outputs = layer_norm(features, reuse=reuse, name='layer_norm_1')
            outputs = tf.nn.dropout(outputs, rate=drop_rate)
            outputs = top_self_attention(outputs, dim=dim, num_heads=num_heads, mask=mask, drop_rate=attn_drop,
                                         reuse=reuse)
            residual = tf.nn.dropout(outputs, rate=drop_rate) + features
            # feed forward layer
            outputs = layer_norm(residual, reuse=reuse, name='layer_norm_2')
            outputs = tf.nn.dropout(outputs, rate=drop_rate)
            outputs = conv1d(outputs, dim=dim, use_bias=True, activation=None, reuse=reuse, name='dense')
            outputs = tf.nn.dropout(outputs, rate=drop_rate) + residual
        return outputs


def conditioned_predictor(inputs, dim, num_heads, max_pos_len, mask, drop_rate, attn_drop, activation=tf.nn.relu,
                          reuse=None, name="conditioned_predictor"):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        start_features = feature_encoder(inputs, dim=dim, num_heads=num_heads, max_pos_len=max_pos_len, mask=mask,
                                         drop_rate=drop_rate, attn_drop=attn_drop, reuse=False, name="feature_encoder")
        end_features = feature_encoder(start_features, dim=dim, num_heads=num_heads, max_pos_len=max_pos_len, mask=mask,
                                       drop_rate=drop_rate, attn_drop=attn_drop, reuse=True, name="feature_encoder")
        start_features = layer_norm(start_features, reuse=False, name='start_layer_norm')
        end_features = layer_norm(end_features, reuse=False, name='end_layer_norm')
        start_features = conv1d(tf.concat([start_features, inputs], axis=-1), dim=dim, use_bias=True,
                                reuse=False, activation=activation, name="start_hidden")
        end_features = conv1d(tf.concat([end_features, inputs], axis=-1), dim=dim, use_bias=True, reuse=False,
                              activation=activation, name="end_hidden")
        start_logits = conv1d(start_features, dim=1, use_bias=True, reuse=reuse, name="start_dense")
        end_logits = conv1d(end_features, dim=1, use_bias=True, reuse=reuse, name="end_dense")
        start_logits = tf.squeeze(start_logits, axis=-1)  # shape = (batch_size, seq_length)
        end_logits = tf.squeeze(end_logits, axis=-1)  # shape = (batch_size, seq_length)
        return start_logits, end_logits
