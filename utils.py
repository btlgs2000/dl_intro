import random
import matplotlib.pyplot as plt
import numpy as np

def display_samples(X, y, y_hat=None, y_prob=None, n_rows=4, n_cols=4, label=None, label_names=None, figsize=(15, 10)):
    choosable_idxs = range(y.size) if label is None else np.where(y == label)[0].tolist()
    idxs = random.sample(population=choosable_idxs, k=n_rows*n_cols)
    true_labels = y[idxs]
    pred_labels = None if y_hat is None else y_hat[idxs]
    probs = None if y_prob is None else y_prob[idxs]
    n_cols = n_cols if y_prob is None else n_cols*2
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            n = j + i*n_cols if y_prob is None else j//2 + i*(n_cols//2)
            ax = axes[i][j]
            if y_prob is None or j%2 == 0:
                ax.set_xticks([])
                ax.set_yticks([])
                label = true_labels[n] if y_hat is None else f't={true_labels[n]},p={pred_labels[n]}'
                ax.set_title(label)
                ax.imshow(X[idxs[n]], cmap='gray')
            else:
                ax.bar(range(10), probs[n], color = ['b' if i == true_labels[n] else 'r' for i in range(10)])