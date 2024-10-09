import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_assignment(
        vector, topology=None, values=False, filename=None, show=True, figsize=(15, 10), title=None
):
    plt.figure(figsize=figsize)
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under(color='white')
    
    cmap_reverse = plt.cm.viridis_r
    cmap_reverse.set_under(color='black')
    p = plt.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')
#     p.set_rasterized(False)

    if values:
        thresh = vector.max() / 2.
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] == -1:
                continue
            else:
                text = '{:.0f}'.format(vector[i, j])
            color = cmap_reverse(vector[i, j] / vector.max())
            diff_color = np.sum(np.array(color) - np.array(cmap(vector[i, j] / vector.max())))
            if np.abs(diff_color) < 0.1:
                red = max(color[0]+0.5, 1.)
                green = max(color[1]-0.5, 0.)
                blue = max(color[2]-0.5, 0.)
                color = (red, blue, green)
#             print(i, j, vector[i, j], diff_color)
            plt.text(j + 0.5, i + 0.5, text,
                     horizontalalignment="center", verticalalignment='center',
                     color=color)

    plt.xlabel('Frequency slot')
    plt.ylabel('Link')
    if title is not None:
        plt.title(title)
#     plt.yticks([topology.edges[link]['id']+.5 for link in topology.edges()], [link[0] + '-' + link[1] for link in topology.edges()])
    if topology is not None:
        plt.yticks([topology.edges[link]['id']+.5 for link in topology.edges()], [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])
    plt.colorbar()
    plt.tight_layout()
    plt.xticks([x+0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()