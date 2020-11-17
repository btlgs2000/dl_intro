import random
import matplotlib.pyplot as plt
import numpy as np

def display_samples(X, y, y_hat=None, y_prob=None, n_rows=4, n_cols=4, label=None, label_names=None, figsize=(15, 10)):
    ''' Mostra una griglia con le immagini di un dataset
    
    args
    ----
    X - pool delle immagini tra cui scegliere quelle da visualizzare nella griglia
    y - label corrispondenti (non one hot)
    y_hat - label predette dal modello (non one hot)
    y_prob - probabilità attribuite dal modello alle varie classi
    n_rows - numero di righe della griglia
    n_cols - numero di colonne della griglia
    label - se None mostra immagini qualsiasi, altrimenti quelle con una specifica label
    '''
    choosable_idxs = range(y.size) if label is None else np.where(y == label)[0].tolist() # indici tra cui scegliere
    idxs = random.sample(population=choosable_idxs, k=n_rows*n_cols) # indici delle immagini da mostrare
    true_labels = y[idxs] # label delle immagini scelte
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
                
def take_test_samples_idxs(y_prob, y_true, is_correct, most_confident, k):
    ''' Ritorna gli indici di k immagini in base ai criteri specificati

    args
    ----
    y_prob - probabilità attribuite dal modello alla varie classi
    y_true - label degli item (non one hot)
    is_correct - True per scegliere le immagini clssificate correttamente
    most_confident - True per scegliere le immagini su cui il modello è più confidente (a torto o a ragione)
    k - numero di immagini da selezionare

    rets
    ----
    idxs - array di indici
    '''
    y_hat = np.argmax(y_prob, axis=1)
    choosable_idxs = np.nonzero(y_hat == y_true)[0] if is_correct else np.nonzero(y_hat != y_true)[0]
    # probabilità massime attribuite dal modello
    # sono le probabilità attribuite alle label predette 
    y_hat_prob = np.amax(y_prob, axis=1)
    sorted_idxs = np.argsort(y_hat_prob)
    if most_confident:
        sorted_idxs = sorted_idxs[::-1]
    return sorted_idxs[np.isin(sorted_idxs, choosable_idxs)][:k]
