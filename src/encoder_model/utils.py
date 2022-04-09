import csv

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

from src import grid_search_config as hp_config
from src import shared_config as shared_config
from src.encoder_model.encoder_utils import CustomSchedule


def f1_score(y_true, y_pred):
    """
    Calculate and return weighted F1 score.
    From https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    ground_positives = K.sum(y_true, axis=0) + K.epsilon()

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1


def create_hyperparams(cli_args):
    """Instantiate and return a dictionary of hyperparameters passed via argparse."""
    h_params = {
        hp_config.HP_BATCH_SIZE: cli_args.batch_size,
        hp_config.HP_DROPOUT: cli_args.dropout_rate,
        hp_config.HP_D_FF_1: cli_args.dimension_ff_1,
        hp_config.HP_D_FF_2: cli_args.dimension_ff_2,
        hp_config.HP_LR: cli_args.learning_rate,
        hp_config.HP_EPOCHS: cli_args.epochs,
        hp_config.HP_POOL_TYPE: cli_args.pool_type,
        hp_config.HP_MAX_POS_ENCODING: cli_args.max_pos_encoding,
        hp_config.HP_D_MODEL: cli_args.d_model,
        hp_config.HP_KEY_CHAIN: cli_args.key_chain,
        hp_config.HP_REL_POS_ENC: cli_args.relative,
        hp_config.HP_WARM_UP_STEPS: cli_args.warm_up_steps,
        hp_config.HP_MAX_DISTANCE: cli_args.max_distance,
        hp_config.HP_D_MODEL_MULT: cli_args.d_model_multiplier,
        hp_config.HP_LR_SCHED: cli_args.learning_rate_scheduler,
        hp_config.HP_CHUNK_SIZE: shared_config.CHUNK_SIZE,
        hp_config.HP_DG1_NEL: cli_args.degree_1_nel,
        hp_config.HP_DG2_NEL: cli_args.degree_2_nel,
        hp_config.HP_QUALITY_NEL: cli_args.quality_nel,
        hp_config.HP_INVERSION_NEL: cli_args.inversion_nel,
        hp_config.HP_KEY_NEL: cli_args.key_nel,
        hp_config.HP_ROOT_NEL: cli_args.root_nel,
        hp_config.HP_DG1_NAH: cli_args.degree_1_nah,
        hp_config.HP_DG2_NAH: cli_args.degree_2_nah,
        hp_config.HP_QUALITY_NAH: cli_args.quality_nah,
        hp_config.HP_INVERSION_NAH: cli_args.inversion_nah,
        hp_config.HP_KEY_NAH: cli_args.key_nah,
        hp_config.HP_ROOT_NAH: cli_args.root_nah,
        hp_config.HP_DG1_REL: cli_args.degree_1_relative,
        hp_config.HP_DG2_REL: cli_args.degree_2_relative,
        hp_config.HP_QUALITY_REL: cli_args.quality_relative,
        hp_config.HP_INVERSION_REL: cli_args.inversion_relative,
        hp_config.HP_KEY_REL: cli_args.key_relative,
        hp_config.HP_ROOT_REL: cli_args.root_relative,
        hp_config.HP_QUALITY_CHAIN: cli_args.quality_chain,
        hp_config.HP_BETA_ONE: cli_args.beta_1,
        hp_config.HP_BETA_TWO: cli_args.beta_2,
        hp_config.HP_EPSILON: cli_args.epsilon,
        hp_config.HP_LABEL_SMOOTHING: cli_args.label_smoothing_eps,
        hp_config.HP_KEY_DROPOUT_RATE: cli_args.key_dropout
    }
    return h_params


def register_metrics(accuracies):
    """Register each metric prior to logging to TensorBoard."""
    tf.summary.scalar(hp_config.METRIC_KEY_ACCURACY, accuracies['key'], step=1)
    tf.summary.scalar(hp_config.METRIC_DG1_ACCURACY, accuracies['degree 1'], step=1)
    tf.summary.scalar(hp_config.METRIC_DG2_ACCURACY, accuracies['degree 2'], step=1)
    tf.summary.scalar(hp_config.METRIC_QUALITY_ACCURACY, accuracies['quality'], step=1)
    tf.summary.scalar(hp_config.METRIC_INV_ACCURACY, accuracies['inversion'], step=1)
    tf.summary.scalar(hp_config.METRIC_ROOT_ACCURACY, accuracies['root'], step=1)
    tf.summary.scalar(hp_config.METRIC_DEGREE_ACCURACY, accuracies['degree'], step=1)
    tf.summary.scalar(hp_config.METRIC_SECONDARY_ACCURACY, accuracies['secondary'], step=1)
    tf.summary.scalar(hp_config.METRIC_ROMAN_ACCURACY, accuracies['roman'], step=1)
    tf.summary.scalar(hp_config.METRIC_ROMAN_INV_ACCURACY, accuracies['roman + inv'], step=1)
    tf.summary.scalar(hp_config.METRIC_ROOT_COHERENCE_ACCURACY, accuracies['root coherence'], step=1)
    tf.summary.scalar(hp_config.METRIC_D7_NO_INC_ACCURACY, accuracies['d7 no inv'], step=1)


def print_cli_args(cli_args):
    """Iterate through command line arguments and print the hyperparameter and its value."""
    print('-' * 50)  # print out the command line arguments
    print('CLI ARGS:')
    for arg in vars(cli_args):
        print(arg, getattr(cli_args, arg))
    print('-' * 50)


def write_hparams_csv(cli_args, model_name):
    """Write a .csv file containing hyperparameter options for each model run."""
    with open('logs/hparams/' + model_name + '_hparams.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)  # save down a copy of the hyperparameters in a csv
        for arg in vars(cli_args):
            writer.writerow([arg, getattr(cli_args, arg)])


def print_batch_attributes(batch_size, h_params, num_features, seq_length):
    """Print attributes specific to the batch prior to training the model."""
    print("-" * 50)
    print(f"BATCH SIZE: {batch_size}")
    print(f"SEQUENCE LENGTH: {seq_length}")
    print(f"NUM FEATURES: {num_features}")
    print(f"FRAMES PER TIME STEP: {seq_length / h_params[hp_config.HP_CHUNK_SIZE]}")
    print("-" * 50)


def get_lr_scheduler(h_params):
    """
    Either return an exponential decay learning rate scheduler
    from TensorFlow, or a CustomSchedule object constructed to
    match the learning rate decay used in the Transformer model.
    """
    if h_params[hp_config.HP_LR_SCHED] == "exponential":
        lr_schedule = ExponentialDecay(
            initial_learning_rate=h_params[hp_config.HP_LR],
            decay_steps=1000,
            decay_rate=0.90,
            staircase=False,
        )
    else:
        lr_schedule = CustomSchedule(
            d_model=h_params[hp_config.HP_D_MODEL],
            d_model_mult=h_params[hp_config.HP_D_MODEL_MULT],
            warmup_steps=h_params[hp_config.HP_WARM_UP_STEPS],
        )
    return lr_schedule


def get_batch_attributes(train_data):
    """Get shape of one item in the batch to print out its attributes."""
    seq_length = [x[0][0].shape[1] for x in train_data.take(1)][0]
    batch_size = [x[0][0].shape[0] for x in train_data.take(1)][0]
    num_features = [x[0][0].shape[2] for x in train_data.take(1)][0]
    return batch_size, num_features, seq_length


def get_callbacks(h_params, model_folder, model_name, model_path, watch_metric):
    """Construct callbacks and return in an array to pass to the model."""
    callbacks = [
        EarlyStopping(
            monitor=watch_metric,
            patience=5,
            verbose=1,
            restore_best_weights=True,
            min_delta=0.001,
            mode="min",
        ),
        TensorBoard(log_dir=model_folder),
        hp.KerasCallback("logs/hparam_tuning/" + model_name, h_params, trial_id=model_name),
        ModelCheckpoint(filepath=model_path, save_best_only=True, monitor=watch_metric),
    ]
    return callbacks
