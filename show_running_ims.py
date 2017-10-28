import numpy as np
import matplotlib.pyplot as plt


num_columns = 4
num_imgs_plot = 3

imgs = np.load('running_imgs.npy')
labs = np.load('running_labs.npy')
predictions = np.load('running_preds.npy')
logits = np.load('running_logits.npy')

for i in range(num_imgs_plot):

    plt.subplot(num_imgs_plot, num_columns,
                i * num_columns + 1)
    im = plt.imshow(imgs[i])
    plt.colorbar(im)
    plt.subplot(num_imgs_plot, num_columns,
                i * num_columns + 2)
    lab = plt.imshow(labs[i])
    plt.colorbar(lab)
    plt.subplot(num_imgs_plot, num_columns,
                i * num_columns + 3)
    logits_out = plt.imshow(logits[i])
    plt.colorbar(logits_out)
    plt.subplot(num_imgs_plot, num_columns,
                i * num_columns + 4)
    preds = plt.imshow(predictions[i])
    plt.colorbar(preds)
plt.show()
