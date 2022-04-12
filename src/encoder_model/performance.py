"""
Code related to the analysis of the RN model's outputs.
"""

from collections import OrderedDict

import warnings
import matplotlib as mpl

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

from src.shared_config import FEATURES, KEYS_SPELLING, KEYS_PITCH, QUALITY, PITCH_FIFTHS, NOTES

warnings.filterwarnings("ignore")
# plotting configurations
sns.set()

rc = {
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "font.family": ["serif"],
    "grid.color": "gainsboro",
    "grid.linestyle": "-",
    "patch.edgecolor": "none",
}
sns.set_style(rc=rc)
mpl.rcParams["figure.edgecolor"] = "black"
mpl.rcParams["axes.linewidth"] = 0.5


def plot_confusion_matrices(task_pred_dicts) -> None:
    """
    Given a dictionary full of prediction metrics, iterate through each task
    and create confusion matrices for them. Saves the figure to the working directory
    in a .png format.
    Args:
        task_pred_dicts (dict): Dictionary containing predictions and actual values.

    Returns:
        None
    """
    idx_to_labels = get_pred_label_opts(
        pitch_spelling=True, return_dict=True
    )
    cmap = sns.cubehelix_palette(
        start=0.5, rot=-0.75, as_cmap=True
    )
    for task_idx, task in enumerate(
        task_pred_dicts.keys()
    ):  # for each task
        labels = list(idx_to_labels[task_idx].values())
        task_cf_mat = confusion_matrix(
            task_pred_dicts[task]["actual"],
            task_pred_dicts[task]["predictions"],
            labels=labels,
        )
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        fmt = lambda x, pos: "{:,.0f}".format(x)
        sns.heatmap(
            task_cf_mat,
            annot=False,
            ax=ax,
            yticklabels=labels,
            xticklabels=labels,
            cmap=cmap,
            lw=0.5,
            cbar_kws={"format": FuncFormatter(fmt)},
        )  # add labels to heatmap axes
        ax.set_title(
            f"Confusion Matrix for {task.title()} Predictions",
            fontdict={
                "fontsize": 14,
                "fontweight": "medium",
            },
        )
        fig.tight_layout()
        plt.savefig(f"{task}_confusion_matrix.png", dpi=200)
        plt.show()


def get_f1_prec_rec(task_pred_dicts) -> dict:
    """
    Calculate weighted f1, recall, and precision scores for each task.
    A dictionary of dictionaries is constructed, and the metrics are added in a
    {'task': {'f1': 0.00, 'recall': 0.00, 'precision': 0.00}} format.
    Args:
        task_pred_dicts (dict):

    Returns:
        Dict of dict containing performance metrics.
    """
    task_perf_metrics = {k: dict() for k in FEATURES}
    for task_idx, task in enumerate(task_pred_dicts.keys()):
        task_actual = task_pred_dicts[task]["actual"]
        task_preds = task_pred_dicts[task]["predictions"]

        task_f1 = f1_score(
            task_actual,
            task_preds,
            labels=np.unique(task_preds),
            average="weighted",
        )
        task_recall = recall_score(
            task_actual,
            task_preds,
            labels=np.unique(task_preds),
            average="weighted",
        )
        task_precision = precision_score(
            task_actual,
            task_preds,
            labels=np.unique(task_preds),
            average="weighted",
        )

        task_perf_metrics[task]["f1"] = task_f1
        task_perf_metrics[task]["recall"] = task_recall
        task_perf_metrics[task][
            "precision"
        ] = task_precision

    return task_perf_metrics


def get_pred_label_opts(pitch_spelling, return_dict=False):
    """
    Get the tick_labels for each task, used for plotting. tick_labels
    array was constructed by Micchi et al in their code base and
    has been used here.

    Args:
        pitch_spelling (bool): Determines number of possible keys depending on pitch encoding method.
        return_dict (bool): Indicates if dictionary is constructed to make it easier to access.

    Returns:
        List or list of OrderedDict depending on return_dict parameter.
    """
    tick_labels = [
        KEYS_SPELLING if pitch_spelling else KEYS_PITCH,  # keys
        [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)]
        + [str(x + 1) + '#' for x in range(7)],  # DEGREE 1
        [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)]
        + [str(x + 1) + '#' for x in range(7)],  # DEGREE 2
        QUALITY,  # QUALITY
        [str(x) for x in range(4)],  # INVERSION
        PITCH_FIFTHS if pitch_spelling else NOTES,  #
        PITCH_FIFTHS if pitch_spelling else NOTES,  # ROOT
    ]

    if return_dict:  # optionally return an ordered dict of each task as opposed to an array
        tick_labels = [
            OrderedDict(
                {k: v for k, v in enumerate(tick_labels[index])})
            for index in range(0, 6)
        ]
    return tick_labels


def get_pred_labels(y_true, y_pred):
    label_dicts = get_pred_label_opts(pitch_spelling=True, return_dict=True)
    output = dict()  # create empty output array for each task's prediction labels
    for j in range(len(y_true)):  # for each task

        task = FEATURES[j]
        tmp_preds = np.argmax(y_pred[j], axis=-1)  # get the logits for the preds
        tmp_true = np.argmax(y_true[j], axis=-1)  # same for true labels

        idx_to_label = label_dicts[j]  # get dictionary that translates to label from index
        tmp_pred_labels = np.vectorize(idx_to_label.get)(tmp_preds)  # map the dictionary to both
        tmp_true_labels = np.vectorize(idx_to_label.get)(tmp_true)

        tmp_output = {'predictions': tmp_pred_labels, 'actual': tmp_true_labels}

        output[task] = tmp_output
    return output