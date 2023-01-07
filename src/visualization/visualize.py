import matplotlib.pyplot as plt


def view_slice(slice, title='', gray=False):
    plt.title(title)

    cmap = None    
    vmin = None
    vmax = None
    
    if gray:
        cmap = 'gray'

    plt.imshow(slice, cmap=cmap)
    plt.show()