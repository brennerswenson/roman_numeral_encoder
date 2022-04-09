import tensorflow as tf


def get_closest_split(n, close_to):
    """From https://stackoverflow.com/questions/57154745/how-to-find-nearest-divisor-to-given-value-with-modulo-zero"""
    all_divisors = get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if close_to < val:
            if ix == 0:
                return val
            if (val - close_to) > (close_to - all_divisors[ix - 1]):
                return all_divisors[ix - 1]
            return val


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.

    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1145

    The first of these two dimensions is n.

    Args:
      x: a Tensor with shape [..., m]
      n: an integer.

    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)  # modified from source
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def split_heads(x, num_heads):
    """
    Split channels (dimension 2) into multiple heads (becomes dimension 1).
    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1198

    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer

    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
    """Inverse of split_heads.
    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1242
    Args:
      x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

    Returns:
      a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.

    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1165
    Args:
      x: a Tensor with shape [..., a, b]

    Returns:
      a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


def get_divisors(n, res=None):
    """From https://stackoverflow.com/questions/57154745/how-to-find-nearest-divisor-to-given-value-with-modulo-zero"""
    res = res or []
    i = 1
    while i <= n:
        if n % i == 0:
            res.append(i),
        i = i + 1
    return res


def shape_list(x):
    """
    Return list of dims, statically where possible.

    From https://github.com/tensorflow/tensor2tensor/blob/2a33b152d7835af66a6d20afe7961751047e28dd/tensor2tensor/layers/common_layers.py#L2815

    """
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def attention_image_summary(attn, image_shapes=None):
    """Compute color image summary.

    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1283

    Args:
      attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
      image_shapes: optional tuple of integer scalars.
        If the query positions and memory positions represent the
        pixels of flattened images, then pass in their dimensions:
          (query_rows, query_cols, memory_rows, memory_cols).
        If the query positions and memory positions represent the
        pixels x channels of flattened images, then pass in their dimensions:
          (query_rows, query_cols, query_channels,
           memory_rows, memory_cols, memory_channels).
    """
    attn = tf.cast(attn, tf.float32)
    num_heads = shape_list(attn)[1]  # modified from source
    # [batch, query_length, memory_length, num_heads]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
    image = split_last_dimension(image, 3)
    image = tf.reduce_max(image, 4)
    if image_shapes is not None:
        if len(image_shapes) == 4:
            q_rows, q_cols, m_rows, m_cols = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
            image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
            image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
        else:
            assert len(image_shapes) == 6
            q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3])
            image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
            image = tf.reshape(
                image,
                [-1, q_rows * m_rows * q_channnels, q_cols * m_cols * m_channels, 3],
            )
    tf.summary.image("attention", image, max_outputs=1)


def top_kth_iterative(x, k):
    """Compute the k-th top element of x on the last axis iteratively.

    From https://github.com/tensorflow/tensor2tensor/blob/2a33b152d7835af66a6d20afe7961751047e28dd/tensor2tensor/layers/common_layers.py#L3270

    This assumes values in x are non-negative, rescale if needed.
    It is often faster than tf.nn.top_k for small k, especially if k < 30.
    Note: this does not support back-propagation, it stops gradients!

    Args:
      x: a Tensor of non-negative numbers of type float.
      k: a python integer.

    Returns:
      a float tensor of the same shape as x but with 1 on the last axis
      that contains the k-th largest number in x.
    """
    # The iterative computation is as follows:
    #
    # cur_x = x
    # for _ in range(k):
    #   top_x = maximum of elements of cur_x on the last axis
    #   cur_x = cur_x where cur_x < top_x and 0 everywhere else (top elements)
    #
    # We encode this computation in a TF graph using tf.foldl, so the inner
    # part of the above loop is called "next_x" and tf.foldl does the loop.
    def next_x(cur_x, _):
        top_x = tf.reduce_max(cur_x, axis=-1, keep_dims=True)
        return cur_x * to_float(cur_x < top_x)

    # We only do k-1 steps of the loop and compute the final max separately.
    fin_x = tf.foldl(
        next_x,
        tf.range(k - 1),
        initializer=tf.stop_gradient(x),
        parallel_iterations=2,
        back_prop=False,
    )
    return tf.stop_gradient(tf.reduce_max(fin_x, axis=-1, keep_dims=True))


def to_float(x):
    """
    Cast x to float; created because tf.to_float is deprecated.
    From https://github.com/tensorflow/tensor2tensor/blob/2a33b152d7835af66a6d20afe7961751047e28dd/tensor2tensor/layers/common_layers.py#L97
    """
    return tf.cast(x, tf.float32)