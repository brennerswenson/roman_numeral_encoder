import os
import time
from math import ceil

import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow import enable_eager_execution
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from config import SHUFFLE_BUFFER, BATCH_SIZE, EPOCHS, VALID_BATCH_SIZE, VALID_STEPS, INPUT_TYPES, DATA_FOLDER
from load_data import create_tfrecords_dataset
from model import create_model, TimeOut
from utils import setup_tfrecords_paths


def visualize_data(data):
    for x, y in data:
        prs, masks, _, _ = x
        for pr, mask in zip(prs, masks):
            sns.heatmap(pr)
            plt.show()
            plt.plot(mask.numpy())
            plt.show()
    return


def setup_model_paths(exploratory, model_type, input_type):
    os.makedirs('models', exist_ok=True)
    if exploratory:
        enable_eager_execution()
        name = 'temp'
    else:
        i = 0
        name = '_'.join([model_type, input_type, str(i)])
        while name in os.listdir('models'):
            i += 1
            name = '_'.join([model_type, input_type, str(i)])

    folder = os.path.join('models', name)
    os.makedirs(folder, exist_ok=True if exploratory else False)

    return folder, name


timeout = None
exploratory = False
# exploratory = True
# model_type = 'conv_dil'
# model_type = 'conv_gru'
model_type = 'gru'
input_type = 'spelling_complete_cut'
if input_type not in INPUT_TYPES:
    raise ValueError('Choose a valid value for input_type')

if __name__ == '__main__':
    model_folder, model_name = setup_model_paths(exploratory, model_type, input_type)
    model_path = os.path.join(model_folder, model_name + '.h5')
    train_path, valid_path, _ = setup_tfrecords_paths(DATA_FOLDER, input_type)
    train_data = create_tfrecords_dataset(train_path, BATCH_SIZE, SHUFFLE_BUFFER, input_type)
    valid_data = create_tfrecords_dataset(valid_path, VALID_BATCH_SIZE, 1, input_type)
    # visualize_data(train_data)

    n_train = 18_792 if input_type.startswith('pitch') else 15_252  # count_records(train_path)
    train_steps = ceil(n_train / BATCH_SIZE)

    model = create_model(model_name, model_type=model_type, input_type=input_type, derive_root=False)
    model.summary()
    print(model_name)

    callbacks = [
        EarlyStopping(patience=3),
        TensorBoard(log_dir=model_folder),
        ModelCheckpoint(filepath=model_path, save_best_only=True)
    ]
    if timeout is not None:
        t0 = time.time()
        callbacks.append(TimeOut(t0=t0, timeout=timeout))

    # weights = [1., 0.5, 1., 1., 0.5, 2.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
    weights = [1., 1., 1., 1., 1., 1.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
    model.compile(loss='categorical_crossentropy', loss_weights=weights, optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, epochs=EPOCHS, steps_per_epoch=train_steps, validation_data=valid_data,
              validation_steps=VALID_STEPS, callbacks=callbacks)

    # model.save(model_path)