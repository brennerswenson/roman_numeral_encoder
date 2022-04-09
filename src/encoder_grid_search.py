import random
import subprocess
import itertools

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tqdm import tqdm

import src.grid_search_config as hp_config

if __name__ == '__main__':

    num_combos = 1
    for name, param in hp_config.HPARAMS_DICT.items():  # get number of unique combos to iterate through for progress bar
        num_combos *= len(param.domain.values)

    possible_values_arr = [x.domain.values for x in hp_config.HPARAMS_DICT.values()]
    hparam_keys = [x for x in hp_config.HPARAMS_DICT.keys()]
    h_param_arr = list()

    # create all unique combos of hyperparameters
    with tqdm(total=num_combos) as progress_bar:
        for combo in itertools.product(*possible_values_arr):
            h_param_arr.append(dict(zip(hparam_keys, combo)))
            progress_bar.update(1)

    with tf.summary.create_file_writer("logs/hparam_tuning").as_default():  # initialize grid search hparams to record in tensorboard
        hp.hparams_config(
            hparams=[
                hp_config.HP_D_MODEL,
                hp_config.HP_BATCH_SIZE,
                hp_config.HP_DROPOUT,
                hp_config.HP_D_FF_1,
                hp_config.HP_D_FF_2,
                hp_config.HP_LR,
                hp_config.HP_EPOCHS,
                hp_config.HP_POOL_TYPE,
                hp_config.HP_KEY_CHAIN,
                hp_config.HP_REL_POS_ENC,
                hp_config.HP_WARM_UP_STEPS,
                hp_config.HP_MAX_DISTANCE,
                hp_config.HP_D_MODEL_MULT,
                hp_config.HP_MAX_POS_ENCODING,
                hp_config.HP_LR_SCHED,
                hp_config.HP_CHUNK_SIZE,
                hp_config.HP_DG1_NEL,
                hp_config.HP_DG2_NEL,
                hp_config.HP_QUALITY_NEL,
                hp_config.HP_INVERSION_NEL,
                hp_config.HP_KEY_NEL,
                hp_config.HP_ROOT_NEL,
                hp_config.HP_DG1_NAH,
                hp_config.HP_DG2_NAH,
                hp_config.HP_QUALITY_NAH,
                hp_config.HP_INVERSION_NAH,
                hp_config.HP_KEY_NAH,
                hp_config.HP_ROOT_NAH,
                hp_config.HP_DG1_REL,
                hp_config.HP_DG2_REL,
                hp_config.HP_QUALITY_REL,
                hp_config.HP_INVERSION_REL,
                hp_config.HP_KEY_REL,
                hp_config.HP_ROOT_REL,
                hp_config.HP_QUALITY_CHAIN,
                hp_config.HP_EPSILON,
                hp_config.HP_BETA_ONE,
                hp_config.HP_BETA_TWO,
                hp_config.HP_LABEL_SMOOTHING,
                hp_config.HP_KEY_DROPOUT_RATE

            ],
            metrics=[hp.Metric(hp_config.METRIC_KEY_ACCURACY, display_name="key"),
                     hp.Metric(hp_config.METRIC_DG1_ACCURACY, display_name="degree 1"),
                     hp.Metric(hp_config.METRIC_DG2_ACCURACY, display_name="degree 2"),
                     hp.Metric(hp_config.METRIC_QUALITY_ACCURACY, display_name="quality"),
                     hp.Metric(hp_config.METRIC_INV_ACCURACY, display_name="inversion"),
                     hp.Metric(hp_config.METRIC_ROOT_ACCURACY, display_name="root"),
                     hp.Metric(hp_config.METRIC_DEGREE_ACCURACY, display_name="degree"),
                     hp.Metric(hp_config.METRIC_SECONDARY_ACCURACY, display_name="secondary"),
                     hp.Metric(hp_config.METRIC_ROMAN_ACCURACY, display_name="roman"),
                     hp.Metric(hp_config.METRIC_ROMAN_INV_ACCURACY, display_name="roman + inv"),
                     hp.Metric(hp_config.METRIC_ROOT_COHERENCE_ACCURACY, display_name="root coherence"),
                     hp.Metric(hp_config.METRIC_D7_NO_INC_ACCURACY, display_name="d7 no inv")
                     ],
        )

    for _ in tqdm(h_param_arr):
        combo = random.choice(h_param_arr)
        if combo['max_pos_encoding'] < combo['chunk_size']:  # positional encoding needs to be big enough for data
            continue

        # execute training file for this specific hyperparameter combination
        args = f"-b {combo['batch_size']}  -e {combo['epochs']} " \
               f"-lr {combo['learning_rate']} " \
               f"-d {combo['dropout']} -dff1 {combo['dimension_ff_1']} -dff2 {combo['dimension_ff_2'] } " \
               f"-pt {combo['pool_type']} -mpe {combo['max_pos_encoding']} " \
               f"-dm {combo['d_model']} -kc {combo['key_chain']} -wu {combo['warm_up_steps']} -md {combo['max_distance']} " \
               f"-dmm {combo['d_model_mult']}  -ls {combo['lr_scheduler']} -r {combo['rel_pos_enc']} -cs {combo['chunk_size']} " \
               f"-dg1-nel {combo['dg1_nel']} -dg2-nel {combo['dg2_nel']} -q-nel {combo['quality_nel']} -inv-nel {combo['inversion_nel']} " \
               f"-k-nel {combo['key_nel']} -r-nel {combo['root_nel']} " \
               f"-dg1-nah {combo['dg1_nah']} -dg2-nah {combo['dg2_nah']} -q-nah {combo['quality_nah']} -inv-nah {combo['inversion_nah']} " \
               f"-k-nah {combo['key_nah']} -r-nah {combo['root_nah']} " \
               f"-dg1-rel {combo['dg1_rel']} -dg2-rel {combo['dg2_rel']} -q-rel {combo['quality_rel']} -inv-rel {combo['inversion_rel']} " \
               f"-k-rel {combo['key_rel']} -r-rel {combo['root_rel']} -qc {combo['quality_chain']} -lse {combo['label_smoothing_eps']} "\
               f"-b1 {combo['beta_1']} -b2 {combo['beta_2']} -eps {combo['epsilon']} -kd {combo['key_dropout']} "\
               f"--input 4"

        subprocess.call(f"../venv/Scripts/python encoder_train.py {args}")
