import tensorflow as tf

from src.encoder_model.encoder_utils import point_wise_feed_forward_network
from src.encoder_model.multi_head_attention import MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    """
    Used https://www.tensorflow.org/text/tutorials/transformer as a template, heavily modified from source.

    Individual layer in Encoder stack. Multi-headed self-attention component is contained within it.
    """

    def __init__(self, d_model, num_heads, dff, layer_num, relative, max_distance, rate=0.1, **kwargs):
        """

        Args:
            d_model (int): Number of features in the input vector. Used when calculating MHA depth.
            num_heads (int): Number of attention heads to use.
            dff (int): Dimension of Feed Forward layer after attention block.
            layer_num (int): Numeric identifier for each encoder layer in the encoder stack.
            relative (bool): Boolean flag indicating whether or not to use relative or absolute positional encoding within the attention block.
            max_distance (int): Maximum distance in both forward and backward directions that relative positional encoding will utilise when clipping values.
            rate (float): Dropout rate between 0 and 1 that is used in the dropout layer after calculating attention weights, and after the feed forward layer.
        """
        super(EncoderLayer, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.d_model = d_model
        self.layer_num = layer_num
        self.relative = relative
        self.max_distance = max_distance

        # instantiate multi-head attention block
        self.mha = MultiHeadAttention(
            self.d_model,
            self.num_heads,
            self.layer_num,
            relative=relative,
            max_distance=self.max_distance,
            name=f"{self.name}_mha_{self.layer_num}",
            rate=self.rate,
        )
        # construct dense layer with normalization and configure dropout layers
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff, layer_num, name=self.name)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{self.name}_layernorm1_{self.layer_num}")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{self.name}_layernorm2_{self.layer_num}")
        self.dropout1 = tf.keras.layers.Dropout(self.rate, name=f"{self.name}_dropout1_{self.layer_num}")
        self.dropout2 = tf.keras.layers.Dropout(self.rate, name=f"{self.name}_dropout2_{self.layer_num}")

    def get_config(self):
        """
        Necessary to override this method so the model can be saved and reloaded.
        https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
        """
        config = super().get_config().copy()
        config.update(
            {
                "layer_num": self.layer_num,
                "d_model": self.d_model,
                "dff": self.dff,
                "num_heads": self.num_heads,
                "rate": self.rate,
                "relative": self.relative,
            }
        )
        return config

    def call(self, x, is_training=False, mask=None):
        """Method that processes each batch through the model architecture."""
        # get attention scores
        attn_output, _ = self.mha(x, x, x, mask, is_training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=is_training)

        # the MHA output is added to the original input. This is a residual connection.
        out1 = self.layernorm1(x + attn_output, training=is_training)  # (batch_size, input_seq_len, d_model)

        # ffn a few linear layers with ReLU
        ffn_output = self.ffn(out1)  # batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=is_training)

        # another residual connection that allows gradients to flow through network continuously
        out2 = self.layernorm2(out1 + ffn_output, training=is_training)  # (batch_size, input_seq_len, d_model)
        return out2
