import numpy as np
import matplotlib.pyplot as plt

def draw_event_intensity(t_array, name, *f):
    """
    t_array: x
    f: t \mapsto y
    name: figure name
    """
    t_range = np.arange(t_array[0], t_array[-1], 0.05)
    f_array = np.array([f[0](t_) for t_ in t_array])
    fig, ax = plt.subplots()
    for f_ in f:
        f_range = np.array([f_(t_) for t_ in t_range])
        ax.plot(t_range, f_range)

    ax.plot(t_array, f_array, '.')
    ax.set(xlabel='time (s)', ylabel='intensity',title=name)
    ax.grid()
    fig.savefig(name+".png")
    return