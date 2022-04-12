"""
The RN Encoder model is defined in this module, as well as functions for constructing it.
The Encoder class is based on the TensorFlow Transformer tutorial code:
https://www.tensorflow.org/text/tutorials/transformer
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dropout,
)
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, LayerNormalization, Concatenate

import src.grid_search_config as hp_config
from src.encoder_model.encoder_layer import EncoderLayer
from src.encoder_model.encoder_utils import get_pool, positional_encoding


class Encoder(tf.keras.layers.Layer):
    """Uses https://www.tensorflow.org/text/tutorials/transformer as a template, but heavily modified."""

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
        rate=0.1,
        relative=False,
        max_distance=None,
        **kwargs,
    ):
        """
        Construct the larger encoder stack with constituent Encoder layers. During the instantiation process
        an individual EncoderLayer object is instantiated for the num_layers provided as a class argument.

        When this object is called, the input batch is modified by the positional encoding function to
        encode sequential order. This is then passed through each encoder layer, its outputs are updated with
        attention scores and its output returned.
        Args:
            num_layers (int): Number of encoder layers to construct within the larger encoder stack.
            d_model (int): Number of input features.
            num_heads (int): Number of heads to use when calculating self-attention scores.
                This must be true: d_model % num_heads == 0
            dff (int): Dimension of Feed Forward layer after attention block in each encoder layer.
            maximum_position_encoding (int): Determines the size number of time steps of the positional
                encoding array.
            rate (float): Dropout rate between 0 and 1 that is used in the dropout layer after calculating
                attention weights, and after the feed forward layer.
            relative (bool): Boolean flag indicating whether or not to use relative or absolute positional
                encoding within the attention block.
            max_distance (int): Maximum distance in both forward and backward directions that relative
                positional encoding will utilise when clipping values.
            **kwargs:
        """
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_pos_encoding = maximum_position_encoding
        self.rate = rate
        self.dff = dff
        self.relative = relative
        self.max_distance = max_distance

        # get positional encoding vector
        self.pos_encoding = positional_encoding(self.max_pos_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(
                self.d_model,
                self.num_heads,
                self.dff,
                relative=self.relative,
                max_distance=self.max_distance,
                rate=self.rate,
                layer_num=n,
                name=f"{self.name}_enc_{n}",
            )
            for n in range(self.num_layers)
        ]  # instantiate all of the encoder layers in the encoder stack for this task

        self.dropout = tf.keras.layers.Dropout(self.rate, name=f"{self.name}_dropout")

    def get_config(self):
        """# https://stackoverflow.com/questions/58678836/notimplementederror-layers-
        with-arguments-in-init-must-override-get-conf"""
        config = super().get_config().copy()
        config.update(
            {
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "dff": self.dff,
                "num_heads": self.num_heads,
                "rate": self.rate,
                "max_distance": self.max_distance,
                "relative": self.relative,
                "maximum_position_encoding": self.max_pos_encoding,
            }
        )
        return config

    def call(self, x, is_training=False, mask=None):
        """Add positional encoding vector, then pass through each encoder layer."""
        seq_len = tf.shape(x)[1]

        # (batch_size, time_steps, pitches), eg (128, 160, 70)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=is_training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, is_training=is_training, mask=mask)
        return x  # (batch_size, input_seq_len, d_model)


def build_task_model(x, enc, num_classes, task, dff_2,
                     dropout_rate, pool_type, is_training=False, concat_layers=[]):
    """
    Downsamples input data using provided method, passes batch through instantiated encoder applying
    self attention. Optionally concatenates the outputs of other task models to further inform the decision
    of the current task. Applies regularization layers like Dropout and LayerNorm to prevent overfitting
    and gradient explosion. Returns prediction probabilities via softmax activation function.
    Args:
        x: Input batch
        enc (encoder object): Encoder object for an individual Roman numeral task.
        num_classes (int): Number of output classes for the task, determines the shape of the softmax layer.
        task (str): Roman numeral task. e.g., Key, Degree 2
        dff_2 (int): Dimension of the hidden layer between the Encoder stack and output layer. 
        dropout_rate (float): Dropout used in between Encoder and hidden layer.
        pool_type (str): Determines the downsampling method prior to going through the encoder stack.
        is_training (bool): Indicates whether or not to use random weights in regularization layers.
        concat_layers (bool): List of output layers from previous models to include in the
            predictions of the current layer.

    Returns:
        Tensor of dimension (batch_size, num_classes)
    """
    # (batch_size, time_steps, pitches)
    x = get_pool(pool_type=pool_type, task=task)(x)  # (batch_size, input_seq_len)

    if concat_layers:  # add outputs from previous layers if applicable
        for idx, conc_layer in enumerate(concat_layers):
            # optionally add key predictions to degree inputs
            x = Concatenate(name=f"{task}_{idx}_concat")([x, conc_layer])

    x = enc(x, is_training=is_training, mask=None)  # (batch_size, seq_len, pitches)
    x = LayerNormalization(name=f"{task}_mt_ln_1")(x, training=is_training)
    x = Dropout(rate=dropout_rate, name=f"{task}_mt_dropout_1")(x, training=is_training)
    x = Dense(dff_2, activation="relu", name=f"{task}_mt_f1")(x)
    x = LayerNormalization(name=f"{task}_mt_ln_2")(x, training=is_training)
    output = Dense(num_classes, activation="softmax", name=f"{task}_mt_f2")(x)
    return output


def get_encoder_stacks(x, encoder_dict, input_type, dff_2,
                       dropout_rate, pool_type, key_chain, quality_chain, is_training=False):
    """
    Construct the model around the instantiated encoder stacks and relate the task models together.
    Returns the predictions in an array containing the output matrices for each batch per task.

    Args:
        x (Tensor): Input batch
        encoder_dict (dict): Dictionary of encoder objects containing layers of multitask
            attention mechanisms. Dict keys are the individual roman numeral tasks.
        input_type (str): Type of input representation, determines the prediction labels.
        dff_2 (int): Dimension of the hidden layer between the Encoder stack and output layer.
        dropout_rate (float): Dropout used in between task's Encoder and hidden layer.
        pool_type (str): Determines the downsampling method prior to going through the encoder stack.
        key_chain (bool): Boolean indicating if predictions from key task should be passed on to
            all other tasks.
        quality_chain (bool): Boolean indicating if quality predictions are to be concatenated to
            input prior to the inversion and degree task encoders.
        is_training (bool): Indicates if random weights are to be used in regularization layers.
            Defaults to False.

    Returns:
        List of output matrices in order [key, degree_1, degree_2, quality, inversion, root]
    """
    classes_key = 30 if input_type.startswith("spelling") else 24  # Major keys: 0-11, Minor keys: 12-23
    # 7 degrees * 3: regular, diminished, augmented
    classes_degree = 21
    # the twelve notes without enharmonic duplicates
    classes_root = 35 if input_type.startswith("spelling") else 12
    classes_quality = 12  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'Gr+6', 'It+6', 'Fr+6']
    classes_inversion = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)

    # get key prediction
    key = build_task_model(
        x,
        enc=encoder_dict["key"],
        num_classes=classes_key,
        task="key",
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        is_training=is_training,
    )
    concat_layers = [key] if key_chain else []
    root = build_task_model(  # root informed by key predictions
        x,
        enc=encoder_dict["root"],
        num_classes=classes_root,
        task="root",
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        is_training=is_training,
        concat_layers=concat_layers,
    )
    quality = build_task_model(  # quality informed by key predictions, should it be informed by root, too?
        x,
        enc=encoder_dict["quality"],
        num_classes=classes_quality,
        task="quality",
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        is_training=is_training,
        concat_layers=concat_layers,
    )
    concat_layers = concat_layers + [root] if key_chain else concat_layers
    concat_layers = concat_layers + [quality] if quality_chain else concat_layers
    degree_1 = build_task_model(  # degree 1 informed by key, quality, and root predictions
        x,
        enc=encoder_dict["dg1"],
        num_classes=classes_degree,
        task="dg1",
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        concat_layers=concat_layers,
        is_training=is_training,
    )  # encode relationship between secondary dominants somehow?
    degree_2 = build_task_model(  # degree 2 informed by key, quality, and root predictions
        x,
        enc=encoder_dict["dg2"],
        num_classes=classes_degree,
        task="dg2",
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        concat_layers=concat_layers,
        is_training=is_training,
    ) # should be informed by key and quality because only 7th notes have certain inversions
    inversion = build_task_model(
        x,
        enc=encoder_dict["inv"],
        num_classes=classes_inversion,
        task="inversion",
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        is_training=is_training,
        concat_layers=concat_layers,
    )
    return [key, degree_1, degree_2, quality, inversion, root]


def create_rn_model(
    d_model,
    dff_1,
    max_pos_encoding,
    input_type,
    dff_2,
    pool_type,
    key_chain,
    quality_chain,
    max_distance,
    h_params,
    is_training=False,
    dropout_rate=0.15,
):
    """
    Instantiates an Encoder Block for each of the RN component tasks. Model input dimensions can vary
    if key_chain or quality_chain are True. Passes the instantiated Encoder blocks to a function that
    generates predictions for each task and returns them in a list.
    Args:
        d_model (int): Number of features in the input vector. Used when calculating MHA depth.
        dff_1 (int): Dimension of Feed Forward layer after attention block.
        max_pos_encoding (int): Indicates the size of the positional encoding matrix.
            This project used 640 (160 * 4).
        input_type (str): Type of input encoding used for model training. This project uses
            'spelling_complete_cut'.
        dff_2 (int): Dimension of hidden Feed Forward layer prior to softmax activation layer.
        pool_type (str): Type of downsampling method used on the input sequence to reduce
            dimensionality to match output chord frequency.
        key_chain (bool): Boolean flag indicating if the outputs from the Key task are to be
            concatenated to other task inputs.
        quality_chain (bool): Boolean flag indicating if the outputs from the Quality task are
            to be concatenated to other downstream tasks.
        max_distance (int): Maximum distance in both forward and backward directions that
            relative positional encoding will utilise when clipping values.
        h_params (dict): Dictionary of hyperparameters passed via the command line.
        is_training (bool): Indicates if the model is in a training stage or not. Useful for
            dropout layers that use random components.
        dropout_rate (float): Dropout rate to use in all model dropout layers.

    Returns:
        keras.Model object suitable for training the multi-task RN problem.
    """
    notes_input = Input(shape=(None, d_model), name="piano_roll_input")  # (time_steps, pitches)
    empty_mask = Input(shape=(None, 1), name="mask_input")  # (time_steps / 4, 1)
    blank_1 = Input(shape=(None,), name="_1", dtype=tf.string)  # for filename, not needed
    blank_2 = Input(shape=(None,), name="_2")  # for transposition, not needed
    blank_3 = Input(shape=(None,), name="_3")  # for start time, not needed

    dg_quality = d_model + 30 if key_chain else d_model  # chain key predictions 100
    dg_root = d_model + 30 if key_chain else d_model  # 100

    # add key and root to chain (root informed by key, too)  135 or 70
    dg_d_model_inv = d_model + 30 + 35 if key_chain else d_model
    # add quality outputs if also chained 147 or 82
    dg_d_model_inv = dg_d_model_inv + 12 if quality_chain else dg_d_model_inv

    key_enc = Encoder(  # which key the chords relate to
        num_layers=h_params[hp_config.HP_KEY_NEL],
        d_model=d_model,
        num_heads=h_params[hp_config.HP_KEY_NAH],
        dff=dff_1,
        maximum_position_encoding=max_pos_encoding,
        rate=h_params[hp_config.HP_KEY_DROPOUT_RATE],
        relative=bool(h_params[hp_config.HP_KEY_REL]),
        max_distance=max_distance if bool(h_params[hp_config.HP_KEY_REL]) else None,
        name="key_encoder",
    )

    root_enc = Encoder(  # root of the chord, can differ from bass note if inverted
        num_layers=h_params[hp_config.HP_ROOT_NEL],
        d_model=dg_root,
        num_heads=h_params[hp_config.HP_ROOT_NAH],
        dff=dff_1,
        maximum_position_encoding=max_pos_encoding,
        rate=dropout_rate,
        relative=bool(h_params[hp_config.HP_ROOT_REL]),
        max_distance=max_distance if bool(h_params[hp_config.HP_ROOT_REL]) else None,
        name="root_encoder",
    )

    quality_enc = Encoder(  # chord quality, major minor etc
        num_layers=h_params[hp_config.HP_QUALITY_NEL],
        d_model=dg_quality,
        num_heads=h_params[hp_config.HP_QUALITY_NAH],
        dff=dff_1,
        maximum_position_encoding=max_pos_encoding,
        rate=dropout_rate,
        relative=bool(h_params[hp_config.HP_QUALITY_REL]),
        max_distance=max_distance if bool(h_params[hp_config.HP_QUALITY_REL]) else None,
        name="quality_encoder",
    )

    dg1_enc = Encoder(  # secondary dominants
        num_layers=h_params[hp_config.HP_DG1_NEL],
        d_model=dg_d_model_inv,
        num_heads=h_params[hp_config.HP_DG1_NAH],
        dff=dff_1,
        maximum_position_encoding=max_pos_encoding,
        rate=dropout_rate,
        relative=bool(h_params[hp_config.HP_DG1_REL]),
        max_distance=max_distance if bool(h_params[hp_config.HP_DG1_REL]) else None,
        name="dg1_encoder",
    )

    dg2_enc = Encoder(  # chord dg relative to dg1
        num_layers=h_params[hp_config.HP_DG2_NEL],
        d_model=dg_d_model_inv,
        num_heads=h_params[hp_config.HP_DG2_NAH],
        dff=dff_1,
        maximum_position_encoding=max_pos_encoding,
        rate=dropout_rate,
        relative=bool(h_params[hp_config.HP_DG2_REL]),
        max_distance=max_distance if bool(h_params[hp_config.HP_DG2_REL]) else None,
        name="dg2_encoder",
    )

    inv_enc = Encoder(  # chord inversion, 4 possible
        num_layers=h_params[hp_config.HP_INVERSION_NEL],
        d_model=dg_d_model_inv,
        num_heads=h_params[hp_config.HP_INVERSION_NAH],
        dff=dff_1,
        maximum_position_encoding=max_pos_encoding,
        rate=dropout_rate,
        relative=bool(h_params[hp_config.HP_INVERSION_REL]),
        max_distance=max_distance if bool(h_params[hp_config.HP_INVERSION_REL]) else None,
        name="inv_encoder",
    )

    encs_dict = {  # save each encoder to a dictionary to pass to the model building function
        "dg1": dg1_enc,
        "dg2": dg2_enc,
        "quality": quality_enc,
        "inv": inv_enc,
        "key": key_enc,
        "root": root_enc,
    }

    y = get_encoder_stacks(
        notes_input,
        encs_dict,
        input_type=input_type,
        dff_2=dff_2,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        key_chain=key_chain,
        quality_chain=quality_chain,
        is_training=is_training,
    )
    model = Model(inputs=[notes_input, empty_mask, blank_1, blank_2, blank_3], outputs=y)
    return model
