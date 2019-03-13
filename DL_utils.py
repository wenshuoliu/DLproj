"""Some utility functions."""

from itertools import cycle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib
import seaborn as sns

def plot_roc(y_pred, y_true, classes=None, title=None, savefile=None):
    """This function plot the ROC curve and return the AUC"""
    if len(y_pred.shape)==1:
        y_pred = y_pred.reshape(y_pred.shape+(1,))
        y_true = y_true.reshape(y_true.shape+(1,))
   
    n_classes = y_pred.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    if classes is None:
        legends = ['class'+str(j+1) for j in range(n_classes)]
    elif len(classes) == n_classes:
        legends = classes
    else:
        raise ValueError("Number of classes doesn't match labels")    
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    colors = cycle(['darkorange', 'cornflowerblue', 'navy', 'aqua'])    

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of {0} (area = {1:0.4f})'
                 ''.format(legends[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    else:
        plt.title('ROC curves for all classes')
    plt.legend(loc="lower right")
    if savefile:
        plt.savefig(savefile, dpi=300)
    return roc_auc

SEP = "__"  # Avoid common signs like "_".
coolwarm_256 = sns.color_palette("coolwarm", 256)


def to_np(x):
    return x.data.cpu().numpy()


def coolwarm_256_color_map(x):
    assert x >= 0 and x < 256
    return coolwarm_256[int(x)]


def parallel_coordinates(trials,
                         plot_filename=None,
                         scale_transformers=None,
                         color_map=coolwarm_256_color_map,
                         alpha=0.8,
                         beta=0.3):
    """Plots parallel coordinates graph.

    Inputs:
      trials: `list` of `dict` with coordinate names and a `score` str as the keys of
        the dict. The `score` key will be used as the color of the curves and
        all other keys will be used as coordinates. All the dicts in the list are
        required to have the same keys.
      plot_filename: `str` indicates the filename to save the plot. Do not save
        the file if `None`.
      scale_transformers: `dict` of monotone functions that transform the scale
        of coordinates in `trials`.
      color_map: A function turns real values into color codes.
      alpha: Transparency.
      beta: Tradeoff between linear and quadratic/cubic interpolation.
    """
    if scale_transformers is None:
        scale_transformers = {}
    trial = trials[0]
    assert "score" in trial
    keys = sorted(trial.keys())
    # Assert at least 2 coordinates and a `score`
    assert len(keys) > 2
    for t in trials:
        assert sorted(t.keys()) == keys

    # Digitize the values of categorical keys.
    categorical_keys = []
    for key in keys:
        assert type(trial[key]) in [str, float, int, np.int64]
        if type(trial[key]) is str:
            categorical_keys.append(key)
    digit_map = {}
    for key in categorical_keys:
        digit_map[key] = {}
        values = sorted(list(set([t[key] for t in trials])))
        for v in values:
            digit_map[key][v] = len(digit_map[key])

    def _transform(v, key):
        """Transforms the value by digitizing and re-scaling."""
        if key in categorical_keys:
            v = digit_map[key][v]
        if key in scale_transformers:
            v = scale_transformers[key](v)
        return v

    # Align values of each coordinate
    align_min = {}
    align_max = {}
    for key in keys:
        values = [_transform(t[key], key) for t in trials]
        align_min[key] = min(values)
        align_max[key] = max(values)
        if align_max[key] == align_min[key]:
            align_max[key] += 1
            align_min[key] -= 1

    def _align(v, key):
        return (v - align_min[key]) / float(align_max[key] - align_min[key])

    # Plot axeses and ticks
    coordinates = [key for key in keys if key != "score"]
    coordinates = ["score"] + coordinates
    x = range(len(coordinates))
    fig, ax = plt.subplots(figsize=((len(x) + 0.9) * 2, 5))
    ax.set_xlim(-0.2, len(x) - 0.3)
    plt.xticks(x, coordinates)
    plt.xlabel("parameter")
    plt.yticks([], [])
    axis_color = sns.xkcd_palette(["dark grey"])[0]
    for x_i in x:
        coord = coordinates[x_i]
        ax.axvline(x=x_i, color=axis_color, linewidth=1)
        raw_values = sorted(list(set([t[coord] for t in trials])))
        n = len(raw_values)
        if n <= 10:
            index = range(n)
        else:
            index = [int(i * (n - 1) / 5.) for i in range(6)]
        ticks = [raw_values[i] for i in index]
        for tick in ticks:
            y_tick = _align(_transform(tick, coord), coord)
            label = tick
            if type(label) == float:
                label = "%.5g" % label
            elif type(label) == int:
                label = "%d" % label
            ax.text(x_i, y_tick, label, size=15)
            ax.scatter(x_i, y_tick, color=axis_color, s=15)

    # Interpolate and smooth with spline
    x_interpolate = np.linspace(x[0], x[-1], len(x) * 100)
    for t in trials:
        y = []
        for coord in coordinates:
            v = _align(_transform(t[coord], coord), coord)
            y.append(v)
        score = _align(_transform(t["score"], "score"), "score")
        score *= 255
        linear = interp1d(x, y)
        if len(x) == 3:
            interpolation_kind = "quadratic"
        else:
            interpolation_kind = "cubic"
        spline = interp1d(x, y, kind=interpolation_kind)
        y_interpolate = beta * spline(x_interpolate) + (
            1 - beta) * linear(x_interpolate)
        ax.plot(
            x_interpolate,
            y_interpolate,
            color=color_map(score),
            alpha=alpha,
            linewidth=1)
    if plot_filename is not None:
        fig.savefig(plot_filename)