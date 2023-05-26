import numpy as np


def get_mean_std_2D(data):
    """
    :param data: list of np.arrays with data samples of shape (h, w, C)
    :return: mean, std as list of shape (C, )
    """
    # mean: arithmetic mean (sum of the elements along axis divided by number of elements)
    # std: sqrt(mean(abs(x - x.mean())**2))
    mean_list = []
    stds_list = []

    for sample in data:
        m = np.mean(np.mean(sample, axis=0), axis=0)
        s = np.std(np.std(sample, axis=0), axis=0)

        mean_list.append(m)
        stds_list.append(s)

    mean = np.mean(mean_list, axis=0)
    std = np.mean(stds_list, axis=0)
    return mean.tolist(), std.tolist()


def get_min_max_2D(data):
    glob_max = np.array(-10000.0)
    glob_min = np.array(10000.0)
    for d in data:
        cur_max = np.max(d)
        cur_min = np.min(d)
        if cur_max > glob_max:
            glob_max = cur_max
        if cur_min < glob_min:
            glob_min = cur_min
    return glob_min, glob_max


if __name__ == "__main__":
    pass
