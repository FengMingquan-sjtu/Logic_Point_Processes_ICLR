import numpy as np
import matplotlib.pyplot as plt

def draw_event_intensity(t_arrays, name, f):
    """
    t_arrays: list of x
    f: list of functions : t \mapsto y
    name: figure name
    """
    t_min = 1e8
    t_max = 0
    for t_a in t_arrays:
        t_min = min(t_min, t_a[0])
        t_max = max(t_max, t_a[-1])
    t_range = np.arange(t_min, t_max, 0.05)

    fig, ax = plt.subplots()
    for f_ in f:# draw intensity
        f_range = np.array([f_(t_) for t_ in t_range])
        ax.plot(t_range, f_range)

    for t_a in t_arrays: # draw points
        f_array = np.array([f[0](t_) for t_ in t_a])
        ax.plot(t_a, f_array, '.')

    ax.set(xlabel='time (s)', ylabel='intensity',title=name)
    ax.grid()
    fig.savefig(name+".png")
    return