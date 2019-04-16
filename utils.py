import numpy as np
import matplotlib.pyplot as plt

# funcao auxiliar para visualizacao dos filtros
def show_filters(filters):

    fig = plt.figure(figsize=(10, 5))
    for i in range(filters.shape[0]):

        ax = fig.add_subplot(1, filters.shape[0], i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filtro %s' % str(i + 1))
        width, height = filters[i].shape

        for x in range(width):
            for y in range(height):

                ax.annotate("{:.2f}".format(filters[i][x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y] < 0 else 'black')


def show_dataset_batch(data, target, last_row=0):

    images = data.numpy()
    for idx in range(len(images)):

        img = images[idx]  # / 2 + 0.5
        if (idx % 10 == 0):
            fig = plt.figure(figsize=(20, 20))

        pos = (idx - (int(np.floor(idx / 10)) * 10) + 1) if idx >= 10 else idx + 1
        ax = fig.add_subplot(1, 10, pos, xticks=[], yticks=[])
        ax.imshow(np.transpose(img, (1, 2, 0)))
        #ax.set_title(class_names[target[idx]])

    last_row += int(np.floor(len(images) / 10))


def show_dataset(dataloader):

    for data, target in dataloader:
        show_dataset_batch(data, target)

