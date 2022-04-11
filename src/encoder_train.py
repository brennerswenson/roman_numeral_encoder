"""
Script configurable via argparse to construct and train the RN encoder model. This script is called via
the subprocess module by the grid search module. Hyperparameters are defined in the argparse argument list.
"""
import argparse
import gc
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from tensorflow.python.keras.optimizer_v2.adam import Adam

import src.grid_search_config as hp_config
import src.shared_config as shared_config
from src.micchi_et_al.utils.analyse_results import generate_results, analyse_results
from src.encoder_model.encoder_model import create_rn_model
from src.micchi_et_al.data_manipulation.load_data import load_tfrecords_dataset
from src.micchi_et_al.utils.utils import setup_tfrecords_paths, setup_model_paths
from src.encoder_model.utils import f1_score, create_hyperparams, register_metrics, print_cli_args, write_hparams_csv, \
    print_batch_attributes, get_lr_scheduler, get_batch_attributes, get_callbacks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.random.set_seed(11)


def main(cli_args):
    """
    Train and save the model using hyperparameters passed via argparse.
    Args:
        cli_args: Argparse command line argument namespace containing hyperparameter values.

    Returns:
        None
    """
    print_cli_args(cli_args)
    h_params = create_hyperparams(cli_args)
    model_type, input_type = shared_config.MODELS[cli_args.model_idx], shared_config.INPUT_TYPES[cli_args.input_idx]
    tfrecords_dir = os.path.join(shared_config.DATA_FOLDER, f'{h_params[hp_config.HP_CHUNK_SIZE]}_chunk')

    train_path, valid_path = setup_tfrecords_paths(tfrecords_dir, ["train", "valid"], input_type)
    model_folder, model_name = setup_model_paths(shared_config.EXPLORATORY, model_type, input_type)
    model_path = os.path.join(model_folder, model_name + ".h5")
    write_hparams_csv(cli_args, model_name)

    with tf.summary.create_file_writer("logs/hparam_tuning/" + model_name).as_default():
        hp.hparams(
            h_params,
            trial_id=model_name
        )  # record the values used in this trial

        # train model
        accuracies = train_encoder_model(
            h_params,
            train_path,
            valid_path,
            input_type,
            model_path,
            model_folder,
            model_name
        )
        register_metrics(accuracies)


def train_encoder_model(h_params, train_path, valid_path, input_type, model_path, model_folder, model_name):
    """
    Args:
        h_params (dict): Dictionary of hyperparameters passed via the command line.
        train_path (str): Filepath containing training TFRecords.
        valid_path (str): Filepath containing validation TFRecords.
        input_type (str): Type of input encoding used for model training. This project uses 'spelling_complete_cut'.
        model_path (str): Filepath indicating where .h5 model file will be saved.
        model_folder (str): Directory location where TensorBoard runtime events and model will be saved.
        model_name (str): Name of the model to be saved.

    Returns:
        (dict) Dictionary of training accuracies for each task, as well as overall RN accuracies.

    """
    train_data = load_tfrecords_dataset(
        train_path,
        h_params[hp_config.HP_BATCH_SIZE],
        shared_config.SHUFFLE_BUFFER,
        input_type,
        h_params[hp_config.HP_CHUNK_SIZE],
        repeat=False,
    )
    valid_data = load_tfrecords_dataset(
        valid_path,
        h_params[hp_config.HP_BATCH_SIZE],
        1,
        input_type,
        h_params[hp_config.HP_CHUNK_SIZE],
        repeat=False
    )
    input_shape = hp_config.INPUT_TYPE2INPUT_SHAPE[input_type]
    batch_size, num_features, seq_length = get_batch_attributes(train_data)
    print_batch_attributes(batch_size, h_params, num_features, seq_length)

    model = create_rn_model(
        d_model=input_shape,
        dff_1=h_params[hp_config.HP_D_FF_1],
        max_pos_encoding=h_params[hp_config.HP_MAX_POS_ENCODING],
        h_params=h_params,
        input_type=input_type,
        dff_2=h_params[hp_config.HP_D_FF_2],
        pool_type=h_params[hp_config.HP_POOL_TYPE],
        key_chain=h_params[hp_config.HP_KEY_CHAIN],
        quality_chain=h_params[hp_config.HP_QUALITY_CHAIN],
        max_distance=h_params[hp_config.HP_MAX_DISTANCE],
        is_training=True,
    )
    model.summary()
    watch_metric = "val_loss"
    callbacks = get_callbacks(h_params, model_folder, model_name, model_path, watch_metric)
    lr_schedule = get_lr_scheduler(h_params)

    opt = Adam(
        learning_rate=lr_schedule,
        beta_1=h_params[hp_config.HP_BETA_ONE],
        beta_2=h_params[hp_config.HP_BETA_TWO],
        epsilon=h_params[hp_config.HP_EPSILON],
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=h_params[hp_config.HP_LABEL_SMOOTHING]),
        optimizer=opt,
        metrics=["accuracy", f1_score],  # tf.metrics.Recall(), tf.metrics.Precision(),
    )
    try:
        model.fit(
            train_data,
            epochs=h_params[hp_config.HP_EPOCHS],
            validation_data=valid_data,
            callbacks=callbacks,
        )
        model.save(model_path)
        ys_true, ys_pred, info = generate_results(
            shared_config.DATA_FOLDER,
            "",
            "encoder_spelling_bass_cut",
            chunk_size=h_params[hp_config.HP_CHUNK_SIZE],
            dataset="valid",
            verbose=True,
            model=model,
        )
        acc = analyse_results(ys_true, ys_pred)
        print(f"SAVED ACCURACY IS {acc}")
        return acc
    except ResourceExhaustedError:
        print("Model too big!")
        del model  # free up memory
        tf.keras.backend.clear_session()
        gc.collect()
        time.sleep(10)
        gc.collect()
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=250,
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.001,
        type=float,
        help="Max learning rate used during training",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="adam",
        help="Type of optimizer to use during training",
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "-d",
        "--dropout-rate",
        default=0.33,
        type=float,
        help="Dropout rate used on Linear layers of neural network.",
    )
    parser.add_argument(
        "-dff1",
        "--dimension-ff-1",
        default=512,
        type=int,
        help="Dimension of Dense layer in Encoder Layers",
    )
    parser.add_argument(
        "-dm",
        "--d-model",
        default=70,
        type=int,
        help="Num of expected features in the encoder inputs",
    )
    parser.add_argument(
        "-dff2",
        "--dimension-ff-2",
        default=128,
        type=int,
        help="Dimension of Dense layer in Multi Task Networks",
    )
    parser.add_argument(
        "-mpe",
        "--max-pos-encoding",
        default=640,
        type=int,
        help="Maximum positional encoding"
    )
    parser.add_argument(
        "--model",
        default=0,
        dest="model_idx",
        action="store",
        type=int,
        help=f'index to select the model, between 0 and {len(shared_config.MODELS)}, {[f"{n}: {m}" for n, m in enumerate(shared_config.MODELS)]}',
    )
    parser.add_argument(
        "--input",
        default=4,
        dest="input_idx",
        action="store",
        type=int,
        help=f'index to select input type, between 0 and {len(shared_config.INPUT_TYPES)}, {[f"{n}: {m}" for n, m in enumerate(shared_config.INPUT_TYPES)]}',
    )
    parser.add_argument(
        "-pt",
        "--pool-type",
        default='avg',
        type=str,
        help="Type of downsampling method.",
    )
    parser.add_argument(
        "-kc",
        "--key-chain",
        default=1,
        type=int,
        help="Flag indicating if model should use key predictions to make other predictions.",
    )
    parser.add_argument(
        "-r",
        "--relative",
        default=0,
        type=int,
        help="Flag indicating if model should use relative positional encodings.",
    )
    parser.add_argument(
        "-wu",
        "--warm-up-steps",
        default=0,
        type=int,
        help="Number of warm up steps in learning rate scheduler.",
    )
    parser.add_argument(
        "-md",
        "--max-distance",
        default=0,
        type=int,
        help="Maximum distance used when calculating relative positional encoding.",
    )
    parser.add_argument(
        "-dmm",
        "--d-model-multiplier",
        default=1,
        type=int,
        help="Factor to mutiply the model dimension by when calculating learning rate schedule.",
    )
    parser.add_argument(
        "-ls",
        "--learning-rate-scheduler",
        default='exponential',
        type=str,
        choices=['custom', 'exponential'],
        help="Determines which learning rate scheduler to use during training.",
    )
    parser.add_argument(
        "-cs",
        "--chunk-size",
        default=160,
        type=int,
        choices=[40, 80, 160, 320, 640, 1280],
        help="Dimension of each chunk when cutting sonatas in chord time-steps.",
    )
    parser.add_argument(
        "-dg1-nel",
        "--degree-1-nel",
        default=2,
        type=int,
        help="Number of encoder layers for the degree 1 encoder.",
    )
    parser.add_argument(
        "-dg2-nel",
        "--degree-2-nel",
        default=2,
        type=int,
        help="Number of encoder layers for the degree 2 encoder.",
    )
    parser.add_argument(
        "-q-nel",
        "--quality-nel",
        default=2,
        type=int,
        help="Number of encoder layers for the quality encoder.",
    )
    parser.add_argument(
        "-inv-nel",
        "--inversion-nel",
        default=2,
        type=int,
        help="Number of encoder layers for the inversion encoder.",
    )
    parser.add_argument(
        "-k-nel",
        "--key-nel",
        default=5,
        type=int,
        help="Number of encoder layers for the key encoder.",
    )
    parser.add_argument(
        "-r-nel",
        "--root-nel",
        default=3,
        type=int,
        help="Number of encoder layers for the root encoder.",
    )
    parser.add_argument(
        "-dg1-nah",
        "--degree-1-nah",
        default=3,
        type=int,
        help="Number of attention heads for the degree 1 encoder layers.",
    )
    parser.add_argument(
        "-dg2-nah",
        "--degree-2-nah",
        default=3,
        type=int,
        help="Number of attention heads for the degree 2 encoder layers.",
    )
    parser.add_argument(
        "-q-nah",
        "--quality-nah",
        default=4,
        type=int,
        help="Number of attention heads for the quality encoder layers.",
    )
    parser.add_argument(
        "-inv-nah",
        "--inversion-nah",
        default=1,
        type=int,
        help="Number of attention heads for the inversion encoder layers.",
    )
    parser.add_argument(
        "-k-nah",
        "--key-nah",
        default=1,
        type=int,
        help="Number of attention heads for the key encoder layers.",
    )
    parser.add_argument(
        "-r-nah",
        "--root-nah",
        default=1,
        type=int,
        help="Number of attention heads for the root encoder layers.",
    )
    parser.add_argument(
        "-dg1-rel",
        "--degree-1-relative",
        default=0,
        type=int,
        help="Indicates if relative positional encodings are to be used for the degree 1 encoder.",
    )
    parser.add_argument(
        "-dg2-rel",
        "--degree-2-relative",
        default=0,
        type=int,
        help="Indicates if relative positional encodings are to be used for the degree 2 encoder.",
    )
    parser.add_argument(
        "-q-rel",
        "--quality-relative",
        default=0,
        type=int,
        help="Indicates if relative positional encodings are to be used for the quality encoder."
    )
    parser.add_argument(
        "-inv-rel",
        "--inversion-relative",
        default=0,
        type=int,
        help="Indicates if relative positional encodings are to be used for the inversion encoder."
    )
    parser.add_argument(
        "-k-rel",
        "--key-relative",
        default=0,
        type=int,
        help="Indicates if relative positional encodings are to be used for the key encoder."
    )
    parser.add_argument(
        "-r-rel",
        "--root-relative",
        default=0,
        type=int,
        help="Indicates if relative positional encodings are to be used for the key encoder."
    )
    parser.add_argument(
        "-qc",
        "--quality-chain",
        default=1,
        type=int,
        help="Indicates if quality predictions are fed to the degree encoders."
    )
    parser.add_argument(
        "-b1",
        "--beta-1",
        default=0.95,
        type=float,
        help="Beta 1 parameter for Adam optimizer."
    )
    parser.add_argument(
        "-b2",
        "--beta-2",
        default=0.95,
        type=float,
        help="Beta 1 parameter for Adam optimizer."
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        default=1e-9,
        type=float,
        help="epsilon parameter for Adam optimizer."
    )
    parser.add_argument(
        "-lse",
        "--label-smoothing-eps",
        default=0.5275,
        type=float,
        help="epsilon parameter loss function label smooothing."
    )
    parser.add_argument(
        "-kd",
        "--key-dropout",
        default=0.15,
        type=float,
        help="Dropout to use in the key encoder stack."
    )
    args = parser.parse_args()
    main(args)
