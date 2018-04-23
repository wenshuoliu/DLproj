"""Some utility functions."""

from itertools import cycle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(y_pred, y_true):
    """This function plot the ROC curve and return the AUC"""
    if len(y_pred.shape)==1:
        y_pred = y_pred.reshape(y_pred.shape+(1,))
        y_true = y_true.reshape(y_true.shape+(1,))
   
    n_classes = y_pred.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy'])
    legends = ['class'+str(j+1) for j in range(n_classes)]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(legends[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for each class')
    plt.legend(loc="lower right")
    return roc_auc