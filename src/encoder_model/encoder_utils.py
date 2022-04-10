"""
Utils functions for constructing RN Encoder model. Any functions with URLs
in the docstring are not mine.
"""
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import AveragePooling1D, MaxPooling1D

from src.encoder_model.attention import _generate_relative_positions_matrix


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """From https://www.tensorflow.org/text/tutorials/transformer"""

    def __init__(self, d_model, d_model_mult, warmup_steps=2000, name=None):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model_float = tf.cast(self.d_model, tf.float16)

        self.warmup_steps = warmup_steps
        self.name = name
        self.d_model_mult = d_model_mult

    def __call__(self, step):
        step = tf.cast(step, tf.float16)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model_float * self.d_model_mult) * tf.math.minimum(
            arg1, arg2
        )  # Modified from the source, used to be * 4, not 2

    def get_config(self):  # Modified from the source
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }


def layers():
    """
    Get the layers module good for TF 1 and TF 2 work for now.
    From https://github.com/tensorflow/tensor2tensor/blob/2a33b152d7835af66a6d20afe7961751047e28dd/tensor2tensor/layers/common_layers.py#L41
    """
    layers_module = None
    try:
        layers_module = tf.layers
    except AttributeError:
        print("Cannot access tf.layers, trying TF2 layers.")
    try:
        from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

        if tf2.enabled():
            print("Running in V2 mode, using Keras layers.")
            layers_module = tf.keras.layers
    except ImportError:
        pass
    return layers_module


def dense(x, units, **kwargs):
    """
    Identical to layers.dense.

    From https://github.com/tensorflow/tensor2tensor/blob/2a33b152d7835af66a6d20afe7961751047e28dd/tensor2tensor/layers/common_layers.py#L3028
    """
    layer_collection = kwargs.pop("layer_collection", None)
    activations = layers().Dense(units, **kwargs)(x)
    if layer_collection:
        # We need to find the layer parameters using scope name for the layer, so
        # check that the layer is named. Otherwise parameters for different layers
        # may get mixed up.
        layer_name = tf.get_variable_scope().name
        if (not layer_name) or ("name" not in kwargs):
            raise ValueError(
                "Variable scope and layer name cannot be empty. Actual: "
                "variable_scope={}, layer name={}".format(layer_name, kwargs.get("name", None))
            )

        layer_name += "/" + kwargs["name"]
        layer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=layer_name)
        assert layer_params
        if len(layer_params) == 1:
            layer_params = layer_params[0]

        tf.logging.info("Registering dense layer to collection for tensor: {}".format(layer_params))

        x_shape = x.shape.as_list()
        if len(x_shape) == 3:
            # Handle [batch, time, depth] inputs by folding batch and time into
            # one dimension: reshaping inputs to [batchxtime, depth].
            x_2d = tf.reshape(x, [-1, x_shape[2]])
            activations_shape = activations.shape.as_list()
            activations_2d = tf.reshape(activations, [-1, activations_shape[2]])
            layer_collection.register_fully_connected_multi(layer_params, x_2d, activations_2d, num_uses=x_shape[1])
            activations = tf.reshape(activations_2d, activations_shape)
        else:
            layer_collection.register_fully_connected(layer_params, x, activations)
    return activations


def point_wise_feed_forward_network(d_model, dff, n, name=None):
    """From https://www.tensorflow.org/text/tutorials/transformer"""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu", name=f"{name}_enc_ffn_dff_{n}"),  # modified
            tf.keras.layers.Dense(
                d_model, name=f"{name}_enc_ffn_d_model_{n}"
            ),  # modified (batch_size, seq_len, d_model)
        ]
    )


def get_pool(pool_type, task):
    """
    Return applicable pooling layer depending on passed argument.
    Args:
        pool_type (str): Type of downsampling method, either max or avg.
        task (str): Relevant task in network for layer naming purposes.

    Returns: Pooling layer
    """
    if pool_type == "avg":
        return AveragePooling1D(4, 4, padding="same", data_format="channels_last", name=f"{task}_pool")
    else:
        return MaxPooling1D(4, 4, padding="same", data_format="channels_last", name=f"{task}_pool")


def get_angles(pos, i, d_model):
    """
    Calculate angle radians for positional encoding purposes.
    From https://www.tensorflow.org/text/tutorials/transformer
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """From https://www.tensorflow.org/text/tutorials/transformer"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_absolute_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1e4, start_index=0):
    """From https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py"""
    position = tf.cast(tf.range(length) + start_index, tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(
        tf.cast(num_timescales, tf.float32) - 1, 1))
    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.compat.v1.mod(hidden_size, 2)]])
    signal = tf.reshape(signal, [1, length, hidden_size])
    return signal


def get_rel_pos_enc(num_steps, num_units, max_distance=10):
    """
    Generate relative positional encodings.
    From https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1670
    """
    rel_matrix = _generate_relative_positions_matrix(num_steps, num_steps, max_distance)
    size_voc = max_distance * 2 + 1
    embeds_table = tf.squeeze(get_absolute_position_encoding(size_voc, num_units), axis=0)
    embeds = tf.gather(embeds_table, rel_matrix)
    return embeds