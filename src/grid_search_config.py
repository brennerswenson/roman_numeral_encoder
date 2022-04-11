from tensorboard.plugins.hparams import api as hp
from collections import OrderedDict
from src.shared_config import INPUT_TYPE2INPUT_SHAPE

# general model hyperparameters
HP_D_MODEL = hp.HParam('d_model', hp.Discrete([INPUT_TYPE2INPUT_SHAPE['spelling_bass_cut']]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.33]))
HP_D_FF_1 = hp.HParam('d_ff_1', hp.Discrete([512]))
HP_D_FF_2 = hp.HParam('d_ff_2', hp.Discrete([128]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.001]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([250]))
HP_POOL_TYPE = hp.HParam('pool_type', hp.Discrete(['avg']))
HP_KEY_CHAIN = hp.HParam('key_chain', hp.Discrete([1]))
HP_REL_POS_ENC = hp.HParam('rel_pos_enc', hp.Discrete([1]))
HP_WARM_UP_STEPS = hp.HParam('warm_up_steps', hp.Discrete([0]))
HP_MAX_DISTANCE = hp.HParam('max_distance', hp.Discrete([0]))
HP_D_MODEL_MULT = hp.HParam('d_model_mult', hp.Discrete([1]))
HP_MAX_POS_ENCODING = hp.HParam('max_pos_encoding', hp.Discrete([640]))
HP_LR_SCHED = hp.HParam('lr_scheduler', hp.Discrete(['exponential']))
HP_CHUNK_SIZE = hp.HParam('chunk_size', hp.Discrete([160]))
HP_QUALITY_CHAIN = hp.HParam('quality_chain', hp.Discrete([1]))
HP_BETA_ONE = hp.HParam('beta_1', hp.Discrete([0.95]))
HP_BETA_TWO = hp.HParam('beta_2', hp.Discrete([0.95]))
HP_EPSILON = hp.HParam('epsilon', hp.Discrete([1e-9]))
HP_LABEL_SMOOTHING = hp.HParam('label_smoothing_eps', hp.Discrete([0.5275]))  # BEST LABEL SMOOTHING WAS 0.2

# number of encoder layers for each task
HP_DG1_NEL = hp.HParam('dg1_nel', hp.Discrete([2]))
HP_DG2_NEL = hp.HParam('dg2_nel', hp.Discrete([2]))
HP_QUALITY_NEL = hp.HParam('quality_nel', hp.Discrete([2]))
HP_INVERSION_NEL = hp.HParam('inversion_nel', hp.Discrete([2]))
HP_KEY_NEL = hp.HParam('key_nel', hp.Discrete([5]))
HP_ROOT_NEL = hp.HParam('root_nel', hp.Discrete([3]))

# number of attention heads for each task
HP_DG1_NAH = hp.HParam('dg1_nah', hp.Discrete([3]))
HP_DG2_NAH = hp.HParam('dg2_nah', hp.Discrete([3]))  # probably just stick with 3
HP_QUALITY_NAH = hp.HParam('quality_nah', hp.Discrete([4]))
HP_INVERSION_NAH = hp.HParam('inversion_nah', hp.Discrete([1]))
HP_KEY_NAH = hp.HParam('key_nah', hp.Discrete([1]))
HP_ROOT_NAH = hp.HParam('root_nah', hp.Discrete([1]))

# number of attention heads for each task
HP_DG1_REL = hp.HParam('dg1_rel', hp.Discrete([0]))
HP_DG2_REL = hp.HParam('dg2_rel', hp.Discrete([0]))
HP_QUALITY_REL = hp.HParam('quality_rel', hp.Discrete([0]))
HP_INVERSION_REL = hp.HParam('inversion_rel', hp.Discrete([0]))
HP_KEY_REL = hp.HParam('key_rel', hp.Discrete([0]))
HP_ROOT_REL = hp.HParam('root_rel', hp.Discrete([0]))

HP_KEY_DROPOUT_RATE = hp.HParam('key_dropout', hp.Discrete([0.15]))

HPARAMS_DICT = OrderedDict({
    'd_model': HP_D_MODEL,
    'batch_size': HP_BATCH_SIZE,
    'dropout': HP_DROPOUT,
    'dimension_ff_1': HP_D_FF_1,
    'dimension_ff_2': HP_D_FF_2,
    'learning_rate': HP_LR,
    'epochs': HP_EPOCHS,
    'pool_type': HP_POOL_TYPE,
    'key_chain': HP_KEY_CHAIN,
    'rel_pos_enc': HP_REL_POS_ENC,
    'warm_up_steps': HP_WARM_UP_STEPS,
    'max_distance': HP_MAX_DISTANCE,
    'd_model_mult': HP_D_MODEL_MULT,
    'max_pos_encoding': HP_MAX_POS_ENCODING,
    'lr_scheduler': HP_LR_SCHED,
    'chunk_size': HP_CHUNK_SIZE,
    'dg1_nel': HP_DG1_NEL,
    'dg2_nel': HP_DG2_NEL,
    'quality_nel': HP_QUALITY_NEL,
    'inversion_nel': HP_INVERSION_NEL,
    'key_nel': HP_KEY_NEL,
    'root_nel': HP_ROOT_NEL,
    'dg1_nah': HP_DG1_NAH,
    'dg2_nah': HP_DG2_NAH,
    'quality_nah': HP_QUALITY_NAH,
    'inversion_nah': HP_INVERSION_NAH,
    'key_nah': HP_KEY_NAH,
    'root_nah': HP_ROOT_NAH,
    'dg1_rel': HP_DG1_REL,
    'dg2_rel': HP_DG2_REL,
    'quality_rel': HP_QUALITY_REL,
    'inversion_rel': HP_INVERSION_REL,
    'key_rel': HP_KEY_REL,
    'root_rel': HP_ROOT_REL,
    'quality_chain': HP_QUALITY_CHAIN,
    'beta_1': HP_BETA_ONE,
    'beta_2': HP_BETA_TWO,
    'epsilon': HP_EPSILON,
    'label_smoothing_eps': HP_LABEL_SMOOTHING,
    'key_dropout': HP_KEY_DROPOUT_RATE
})

# names for evaluation metrics
METRIC_KEY_ACCURACY = 'key'
METRIC_DG1_ACCURACY = 'degree 1'
METRIC_DG2_ACCURACY = 'degree 2'
METRIC_QUALITY_ACCURACY = 'quality'
METRIC_INV_ACCURACY = 'inversion'
METRIC_ROOT_ACCURACY = 'root'
METRIC_DEGREE_ACCURACY = 'degree'
METRIC_SECONDARY_ACCURACY = 'secondary'
METRIC_ROMAN_ACCURACY = 'roman'
METRIC_ROMAN_INV_ACCURACY = 'roman + inv'
METRIC_ROOT_COHERENCE_ACCURACY = 'root coherence'
METRIC_D7_NO_INC_ACCURACY = 'd7 no inv'
