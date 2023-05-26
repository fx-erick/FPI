import torch


class RandomNoiseToTensor(object):
    """
    Class to add noise (from a standard normal distribution multiplied by 0.1)
    to arbitrary data for data augmentation.
    :param p: (float) probability with which transform is applied to a datapoint
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return x + torch.randn_like(x) * 0.1
        else:
            return x
