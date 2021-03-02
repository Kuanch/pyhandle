import numpy as np
import matplotlib.pyplot as plt


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(preds, probs, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    cpu_images = images.cpu()
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(64, 64))
    for idx in np.arange(16):
        ax = fig.add_subplot(16, 1, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(cpu_images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(preds[idx],
                                                          probs[idx] * 100.0,
                                                          labels[idx]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig
