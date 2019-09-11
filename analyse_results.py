import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from config import BATCH_SIZE, FEATURES, TICK_LABELS, CIRCLE_OF_FIFTH, NOTES, \
    VALID_TFRECORDS, VALID_STEPS, VALID_INDICES
from load_data import create_tfrecords_dataset
from utils import find_root_full_output, decode_results, Q2I


def check_predictions(y_true, y_pred, index):
    """
    Check if the predictions are correct independently for each feature and each timestep.

    :param y_true:
    :param y_pred:
    :param index:
    :return: a boolean vector of dimension [features, samples]
    """
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def visualize_chord_changes(y_true, y_pred, inversions=True):
    if inversions:
        yt = [np.argmax(y[0], axis=-1) for y in y_true[:-1]]
        yp = [np.argmax(y[0], axis=-1) for y in y_pred[:-1]]
    else:
        yt = [np.argmax(y[0], axis=-1) for y in y_true[:-2]]
        yp = [np.argmax(y[0], axis=-1) for y in y_pred[:-2]]
    n = len(yt[0])
    change_true, change_pred = np.zeros(n), np.zeros(n)
    for m in range(n - 1):
        if np.any([y[m + 1] != y[m] for y in yt]):
            change_true[m] = 1
        if np.any([y[m + 1] != y[m] for y in yp]):
            change_pred[m] = 1

    # Plotting the results
    cmap = sns.color_palette(['#d73027', '#f7f7f7', '#3027d7'])
    ax = sns.heatmap([change_true - change_pred], cmap=cmap)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.67, 0., 0.67])
    colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])
    ax.set(ylabel=['change_true', 'change_pred'], xlabel='time',
           title=f"Sonata {VALID_INDICES[i]} - chord consistency " +
                 ('with inversions' if inversions else 'without inversions'))
    plt.show()
    zt = decode_results(yt)
    zp = decode_results(yp)
    wt = [' '.join([zt[0][i], zt[1][i], zt[2][i], zt[3][i]]) for i in range(n)]
    wp = [' '.join([zp[0][i], zp[1][i], zp[2][i], zp[3][i]]) for i in range(n)]
    print("hi")
    return


def visualize_results(mode='probabilities'):
    if mode not in ['probabilities', 'predictions']:
        raise ValueError('mode should be either probabilities or predictions')
    cmap = sns.color_palette(['#d73027', '#f7f7f7', '#3027d7', '#000000']) if mode == 'predictions' else 'RdGy'

    for j in range(6):
        a = test_predict[i][j][0] if j > 0 else test_predict[i][j][0][:, CIRCLE_OF_FIFTH]
        b = test_truth[i][j][0] if j > 0 else test_truth[i][j][0][:, CIRCLE_OF_FIFTH]
        if mode == 'predictions':
            a = (a == np.max(a, axis=-1, keepdims=True))
            x = b - a
            x[b == 1] += 1
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, vmin=-1, vmax=2, yticklabels=TICK_LABELS[j])
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-5 / 8, 1 / 8, 7 / 8, 13 / 8])
            colorbar.set_ticklabels(['False Pos', 'True Neg', 'True Pos', 'False Neg'])
        else:
            x = b - a
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, center=0, vmin=-1, vmax=1, yticklabels=TICK_LABELS[j])
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-1, 0, +1])
            colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])

        ax.set(ylabel=FEATURES[j], xlabel='time',
               title=f"Sonata {VALID_INDICES[i]} - {FEATURES[j]}")
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.show()
        return


def visualize_piano_roll(pr, i):
    ax = sns.heatmap(pr[i].transpose(), vmin=0, vmax=1)
    ax.set(xlabel='time', ylabel='notes',
           title=f"Sonata {VALID_INDICES[i]} - piano roll data")
    plt.show()
    return


def plot_coherence(y_symb, y_func, n_classes, sonata):
    msk = (y_symb != y_func)
    y_symb = np.array(y_symb)[msk]
    y_func = np.array(y_func)[msk]
    c = np.zeros((n_classes, n_classes))
    for i, j in zip(y_symb, y_func):
        c[i, j] += 1
    sns.heatmap(c, xticklabels=NOTES, yticklabels=NOTES)
    plt.title(f"Sonata {sonata} - root prediction")
    plt.xlabel("SYMB")
    plt.ylabel("FUNC")
    plt.show()
    return


test_data = create_tfrecords_dataset(VALID_TFRECORDS, BATCH_SIZE, shuffle_buffer=1, n=24)
test_data_iter = test_data.make_one_shot_iterator()
x, y = test_data_iter.get_next()

mode = 'pitch_class'
# mode = 'midi_number'
# model_name = 'conv_gru'
model_name = 'conv_gru_reduced_' + mode
model_folder = os.path.join('logs', model_name)
model = load_model(os.path.join(model_folder, model_name + '.h5'))
model.summary()

# Retrieve the true labels
piano_rolls = []
test_truth = []
with tf.Session() as sess:
    for i in range(VALID_STEPS):
        data = sess.run([x, y])
        piano_rolls.append(data[0][0][0])  # meaning of zeros: x or y, piano roll or bass, first element of batch
        test_truth.append(data[1])

# visualize_piano_roll(piano_rolls, 0)

# Predict new labels and view the difference
test_predict = []  # It will have shape: [pieces, features, (batch size, length of sonata, feature size)]
for i in range(VALID_STEPS):
    print(f"step {i + 1} out of {VALID_STEPS}")
    temp = model.predict(test_data.skip(i), steps=1, verbose=False)
    test_predict.append(temp)

    # visualize_results(mode='predictions')
    # visualize_results(mode='probabilities')

# for i in range(VALID_STEPS):
#     create_dezrann_annotations(test_truth[i], test_predict[i], n=VALID_INDICES[i], batch_size=BATCH_SIZE,
#                                model_folder=model_folder)

# Calculate accuracy etc.
func_tp, name_tp, root_tp, n_predictions = 0, 0, 0, 0
root_coherence = 0
degree_tp, secondary_tp, secondary_total, d7_tp, d7_total = 0, 0, 0, 0, 0
cp = [[] for _ in range(6)]  # dimensions will be: features, sonatas, frames
tp = np.zeros(6)  # true positives for each separate feature
for i in range(VALID_STEPS):
    y_true, y_pred = test_truth[i], test_predict[i]  # shape: [features], [batch, timestep, classes]
    n_predictions += y_true[0].shape[1]
    visualize_chord_changes(y_true, y_pred, True)
    visualize_chord_changes(y_true, y_pred, False)
    for j in range(6):
        for a in check_predictions(y_true, y_pred, j):
            cp[j].append(a)  # cp has shape (features, batch_size*(i+1), l) where l varies
    # can't vectorialize the next three lines because the length of each sonata l is different
    tp += np.sum(np.array([a[-1] for a in cp]), axis=-1)
    func_tp += np.sum(np.prod(np.array([a[-1] for a in cp[:4]]), axis=0), axis=-1)
    degree_tp += np.sum(np.prod(np.array([a[-1] for a in cp[1:3]]), axis=0), axis=-1)
    secondary_msk = (np.argmax(y_true[1][0], axis=-1) != 0)
    secondary_total += sum(secondary_msk)
    secondary_tp += np.sum(np.prod(np.array([a[-1] for a in cp[1:3]]), axis=0)[secondary_msk], axis=-1)
    root_pred = find_root_full_output(y_pred)
    # plot_coherence(np.argmax(y_pred[5], axis=-1)[0], root_pred, n_classes=12, sonata=VALID_INDICES[i])
    root_coherence += np.sum(root_pred == np.argmax(y_pred[5], axis=-1))
    root_tp += np.sum(root_pred == np.argmax(y_true[5], axis=-1))
    d7_msk = (np.argmax(y_true[3][0], axis=-1) == Q2I['d7'])
    d7_total += sum(d7_msk)
    d7_tp += np.sum(np.prod(np.array([a[-1] for a in cp[:4]]), axis=0)[secondary_msk], axis=-1)

acc = tp / n_predictions
degree_acc = degree_tp / n_predictions
secondary_acc = secondary_tp / secondary_total
func_acc = func_tp / n_predictions
print(f"accuracy for the different items:")
for f, a in zip(FEATURES, acc):
    print(f"{f:10} : {a:.4f}")
print(f'degree     : {degree_tp / n_predictions:.4f}')
print(f'secondary  : {secondary_tp / secondary_total:.4f}')
print(f'func root  : {root_tp / n_predictions:.4f}')
print(f"global functional : {func_acc:.4f}")
print(f'root_coherence: {root_coherence / n_predictions:.4f}')
