import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, LayerNormalization

from src.encoder_model.attention_utils import split_last_dimension, split_heads, combine_heads, \
    shape_list, attention_image_summary, top_kth_iterative


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-headed attention layer based on TensorFlow's Transformer tutorial:
    https://www.tensorflow.org/text/tutorials/transformer

    Uses and combines components relative positional encoding components from from:
    https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py

    Module in a Transformer network that computes the attention weights
    for the input and produces an output vector with encoded information
    on how each token should attend to all other tokens in the sequence.
    """

    def __init__(self, d_model, num_heads, layer_num, relative, rate, max_distance, **kwargs):
        """
        Args:
            d_model (int): Number of features in the input vector. Used when calculating MHA depth.
            num_heads (int): Number of attention heads to use.
            layer_num (int): Numeric identifier for each encoder layer in the encoder stack.
            relative (bool): Boolean flag indicating whether or not to use relative or absolute positional encoding within the attention block.
            rate (float): Dropout rate between 0 and 1 that is used in the dropout layer after calculating attention weights, and after the feed forward layer.
            max_distance (int): Maximum distance in both forward and backward directions that relative positional encoding will utilise when clipping values.
            **kwargs:
        """
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.d_model = d_model
        self.layer_num = layer_num
        self.relative = relative
        self.max_distance = max_distance
        self.mha_name = f"{self.name}_{self.layer_num}"

        assert d_model % num_heads == 0  # necessary that the number of attention heads is a factor of number of features

        self.depth = d_model // self.num_heads  # calculate depth
        self.dropout_rate = rate

        # initialise w matrices
        self.wq = tf.keras.layers.Dense(d_model, name=f"{self.name}_mha_wq_{layer_num}")
        self.wk = tf.keras.layers.Dense(d_model, name=f"{self.name}_mha_wk_{layer_num}")
        self.wv = tf.keras.layers.Dense(d_model, name=f"{self.name}_mha_wv_{layer_num}")

        # output layer for absolute positional encoding
        self.dense = tf.keras.layers.Dense(d_model, name=f"{self.name}_mha_dense_{layer_num}")

        # relative positional encoding layers
        self.dense_rpe = Dense(self.d_model, name=f"dense_rel_pe_{self.layer_num}")  # d_model should be 70
        self.dropout = Dropout(self.dropout_rate)
        self.dense_output = Dense(self.d_model, name=f"dense_O_{self.layer_num}")
        self.layer_norm = LayerNormalization()

    def split_heads(self, x):
        """
          Split channels (dimension 2) into multiple heads (becomes dimension 1).
          From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1198

        Args:
          x: a Tensor with shape [batch, length, channels]

        Returns:
          a Tensor with shape [batch, num_heads, length, channels / num_heads]
        """
        return tf.transpose(split_last_dimension(x, self.num_heads), [0, 2, 1, 3])

    def call(self, v, k, q, mask, is_training=False):
        """
        Method that calculates self attention scores for each input batch.

        The boolean class attribute `self.relative` indicates whether or not to
        use relative positional encoding when calculating attention scores. If equal to False,
        absolute attention is used as it was implemented in the Transformer model.
        If equal to True, relative attention scores are calculated using relative positional
        encoding. Max distance is provided as an instantiation argument of the class.

        Args:
            v (tensor): value must be (batch_size, seq_len_v, depth_v)
            k (tensor): key must be (batch_size, seq_len_k, depth)            k:
            q (tensor): query must be (batch_size, seq_len_q, depth)
            mask (Float tensor): Unused parameter in this application.
            is_training (bool):

        Returns:
            attention_scores, attention_weights
        """

        batch_size = tf.shape(q)[0]

        Q = self.wq(q)  # (batch_size, seq_len, d_model)
        K = self.wk(k)  # (batch_size, seq_len, d_model)
        V = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(V)

        if not self.relative:
            # split q, k, v in to N vectors before applying self attention
            # the split vectors then go through the same attention process individually

            # (batch_size, num_heads, seq_len_v, depth)
            # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
            # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)  # TensorShape([32, 1, 80, 70]), TensorShape([32, 1, 80, 80])
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # TensorShape([32, 80, 1, 70])
            concat_attention = tf.reshape(
                scaled_attention,
                (batch_size, -1, self.d_model)
            )  # TensorShape([32, 80, 70]) concat different attention heads
            output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        else:
            # if relative positional encoding
            total_key_depth = K.get_shape().as_list()[-1]
            total_value_depth = V.get_shape().as_list()[-1]

            q = split_heads(Q, self.num_heads)
            key_depth_per_head = total_key_depth // self.num_heads
            q *= key_depth_per_head**-0.5
            x, attention_weights = dot_product_attention_relative(
                q,
                k,
                v,
                bias=None,
                max_relative_position=self.max_distance,
                dropout_rate=self.dropout_rate,
                image_shapes=None,
                save_weights_to=None,
                make_image_summary=False,
                cache=False,
                allow_memory=False,
                hard_attention_k=0,
                gumbel_noise_weight=0.0,
                name=self.mha_name,
            )
            x = combine_heads(x)
            x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])
            output = self.dense_output(x)
        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
    """
    From https://www.tensorflow.org/text/tutorials/transformer

    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    The key/value/query concepts come from retrieval systems. For example,
    when you type a query to search for some video on Youtube, the search engine
    will map your query against a set of keys (video title, description, etc)
    associated with candidate videos in the database, then present you the best
    matched videos (values).

    The attention operation can be thought of as a retrival process as well,
    so the key/value/query concepts also apply here.

    Args:
    q: query shape == (..., seq_len_q, depth)  #
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    # produce a score matrix (determines how much focus one word has on all other words)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # this is to allow for more stable gradients as multiplying values could have exploding effects
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # TensorShape([32, 1, 80, 80])

    if mask is not None:
        scaled_attention_logits += mask * -1e9  # not utilised

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # TensorShape([32, 1, 80, 80])
    # multiply by the value vector to get the output scores.
    output = tf.matmul(attention_weights, v)  # TensorShape([32, 1, 80, 70])

    return output, attention_weights


def dot_product_attention_relative(
    q,
    k,
    v,
    bias,
    max_relative_position,
    dropout_rate=0.0,
    image_shapes=None,
    save_weights_to=None,
    name=None,
    make_image_summary=True,
    cache=False,
    allow_memory=False,
    hard_attention_k=0,
    gumbel_noise_weight=0.0,
):
    """Calculate relative position-aware dot-product self-attention.

    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1739

    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.

    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_relative_position: an integer specifying the maximum distance between
          inputs that unique position embeddings should be learned for.
      dropout_rate: a floating point number.
      image_shapes: optional tuple of integer scalars.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      name: an optional string.
      make_image_summary: Whether to make an attention image summary.
      cache: whether use cache mode
      allow_memory: whether to assume that recurrent memory is in use. If True,
        the length dimension of k/v/bias may be longer than the queries, and it is
        assumed that the extra memory entries precede the non-memory entries.
      hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
      gumbel_noise_weight: if > 0, apply Gumbel noise with weight
        `gumbel_noise_weight` before picking top-k. This is a no op if
        hard_attention_k <= 0.

    Returns:
      A Tensor.

    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError(
            "Max relative position (%s) should be > 0 when using " "relative self attention." % (max_relative_position)
        )

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape, unless memory is enabled.
    if not cache and not allow_memory:
        q.get_shape().assert_is_compatible_with(k.get_shape())
        q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    depth = k.get_shape().as_list()[3]
    length_k = shape_list(k)[2]  # modified from source
    length_q = shape_list(q)[2] if allow_memory else length_k  # modified from source
    relations_keys = _generate_relative_positions_embeddings(
        length_q,
        length_k,
        depth,
        max_relative_position,
        f"{name}_relative_positions_keys",
        cache=cache,
    )
    relations_values = _generate_relative_positions_embeddings(
        length_q,
        length_k,
        depth,
        max_relative_position,
        f"{name}_relative_positions_values",
        cache=cache,
    )

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
        logits += bias
    weights = tf.nn.softmax(logits, name=f"{name}_attention_weights")
    if hard_attention_k > 0:
        weights = harden_attention_weights(weights, hard_attention_k, gumbel_noise_weight)
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if not tf.compat.v1.get_variable_scope().reuse and make_image_summary:  # modified from source
        attention_image_summary(weights, image_shapes)
    return _relative_attention_inner(weights, v, relations_values, False), weights


def _generate_relative_positions_matrix(length_q, length_k, max_relative_position, cache=False):
    """
    Generates matrix of relative positions between inputs.
    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1670
    """
    if not cache:

        if length_q == length_k:
            range_vec_q = range_vec_k = tf.range(length_q)
        else:
            range_vec_k = tf.range(length_k)
            range_vec_q = range_vec_k[-length_q:]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    else:
        distance_mat = tf.expand_dims(tf.range(-length_k + 1, 1, 1), 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position, max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length_q, length_k, depth, max_relative_position, name, cache=False):
    """
    Generates tensor of size [1 if cache else length_q, length_k, depth].
    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1691
    """

    # Generates embedding for each relative position of dimension depth.
    with tf.compat.v1.variable_scope(f"{name}_embeddings", reuse=tf.compat.v1.AUTO_REUSE):  # modified from source
        relative_positions_matrix = _generate_relative_positions_matrix(length_q, length_k, max_relative_position, cache=cache)
        vocab_size = max_relative_position * 2 + 1
        embeddings_table = tf.compat.v1.get_variable(f"{name}_embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings


def _relative_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.

    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1705

    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
      x: Tensor with shape [batch_size, heads, length or 1, length or depth].
      y: Tensor with shape [batch_size, heads, length or 1, depth].
      z: Tensor with shape [length or 1, length, depth].
      transpose: Whether to transpose inner matrices of y and z. Should be true if
          last dimension of x is depth, not length.

    Returns:
      A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length or 1, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    # x_t is [length or 1, batch_size, heads, length or depth]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    # x_t_r is [length or 1, batch_size * heads, length or depth]
    x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
    # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
    # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
    # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t


def harden_attention_weights(weights, k, gumbel_noise_weight):
    """
    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1582
    Make attention weights non-0 only on the top k ones.
    """
    if gumbel_noise_weight > 0.0:
        gumbel_noise = -tf.log(-tf.log(tf.random_uniform(tf.shape(weights), minval=1e-5, maxval=1 - 1e-5)))
        weights += gumbel_noise * gumbel_noise_weight

    # Subtract the top-kth weight and zero-out all lower ones.
    # Note that currently in case of numerical ties it will retain more
    # than k elements. In the future, we may want to avoid this.
    weights -= top_kth_iterative(weights, k)
    weights = tf.nn.relu(weights)
    # Re-normalize the weights.
    weights_sum = tf.reduce_sum(weights, axis=-1, keep_dims=True)
    weights_sum = tf.maximum(weights_sum, 1e-6)  # Avoid division by 0.
    weights /= weights_sum
    return weights
