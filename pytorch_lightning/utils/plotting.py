import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def plot_confusion_matrix(cm,
                          save_path,
                          normalize=False,
                          title='Confusion matrix',
                          ylabel='y',
                          xlabel='x'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from matplotlib import pyplot as plt
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure()
    plt.matshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(save_path)
