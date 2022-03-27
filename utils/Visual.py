import matplotlib.pyplot as plt
import numpy as np


def hist(ax: plt.Axes, img: np.ndarray, title=None, min_base=0, **kwargs):
    arr = img.flatten()
    arr = np.delete(arr, obj=np.where(arr == min_base)[0])

    if 'range' not in kwargs.keys():
        kwargs['range'] = [0, np.percentile(arr, 98)]

    n, bins, patches = ax.hist(arr, **kwargs)

    arr_no_background = arr
    p_90 = np.percentile(arr_no_background, 90)
    ax.vlines(x=[p_90], ymin=0, ymax=1, transform=ax.get_xaxis_transform(), colors="r", label=f"90%: {p_90}")

    ax.set_xlim(bins[0], bins[-1])
    ax.legend(fontsize="large")
    ax.set_title(title, fontsize="x-large")

    return n, bins, patches
    
    
    