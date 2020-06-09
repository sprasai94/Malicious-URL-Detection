import matplotlib.pyplot as plt
import numpy as np


# plot accuracy and loss vs epochs
def plot_learning_curves(history):
    # plot loss vs epochs
    plt.subplot(211)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy vs epochs
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()

    plt.show()
    plt.close()


def find_accuracy(test_data, predictions):
    correct = 0
    # last element of test set is real classification,
    # so making list of true classification
    # actual = [instances[-1] for instances in test_data]
    # comparing each values in list if they are similar
    actual = test_data
    for x, y in zip(actual, predictions):
        if x == y:
            correct += 1
    accuracy = (correct/float(len(actual))) * 100
    return accuracy


# Generate a confusion matrix from predicted and actual classification
def find_confusion_matrix(p, a):
    num_of_classes = 2

    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    for i in range(len(a)):
        confusion_matrix[p[i]][a[i]] += 1
    # print('confusion matrix:', confusion_matrix)
    return confusion_matrix


# Function to plot the confusion matrix
def plotConfusionMatrix(test_set, y_pred, cm,  normalize=True, title=None, cmap = None, plot = True):
    classes = ["benign", "malicious"]

    #print('Confusion matrix (without normalization):')
    #print(classes)
    #print(cm)
    if cmap is None:
        cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if plot:
        plt.show()

