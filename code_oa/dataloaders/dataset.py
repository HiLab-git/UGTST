# -*- coding: utf-8 -*-

from logging import root
import os
from scipy import ndimage
import torch
import random
import h5py
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools
from torchvision import transforms
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import skimage

class h5DataSet(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
            active_method=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.active_method = active_method
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "train_stage1" and self.active_method:
            with open(self._base_dir + f"/stage1_slice_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "train_stage2" and self.active_method:
            with open(self._base_dir + f"/all_slice_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "semi_train" and self.active_method:
            with open(self._base_dir + f"/all_slice_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "semi_train":
            with open(self._base_dir + "/all_slice.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "val":
            with open(self._base_dir + "/vallist.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train" or "train_stage1" or "train_stage2":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "val":
            h5f = h5py.File(self._base_dir + "/{}".format(case), "r")
        elif self.split == "train":
            h5f = h5py.File(self._base_dir + "/slices/{}".format(case), "r")
        elif self.split == "train_stage1":
            h5f = h5py.File(self._base_dir + "/slices/{}".format(case), "r")
        elif self.split == "train_stage2":
            h5f = h5py.File(self._base_dir + "/slices/{}".format(case), "r")
        elif self.split == "semi_train":
            h5f = h5py.File(self._base_dir + "/slices/{}".format(case), "r")

        image = h5f["image"][:]
        label = h5f["label"][:]
        image = image.astype(np.float32)

        label = label.astype(np.uint8)
        sample = {"image": image, "label": label}
        if (self.split == "train" or self.split == "train_stage1" or self.split == "train_stage2" or
                self.split == "semi_train"):
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = case
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def gaussian_noise(image, label):
    mean = 0
    std = 0.05
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    return image, label

def gaussian_blur(image, label):
    std_range = [0, 1]
    std = np.random.uniform(std_range[0], std_range[1])
    image = gaussian_filter(image, std, order=0)
    return image, label

def gammacorrection(image, label):
    gamma_min, gamma_max = 0.7, 1.5
    flip_prob = 0.5
    gamma_c = random.random() * (gamma_max - gamma_min) + gamma_min
    v_min = image.min()
    v_max = image.max()
    if (v_min < v_max):
        image = (image - v_min) / (v_max - v_min)
        if (np.random.uniform() < flip_prob):
            image = 1.0 - image
        image = np.power(image, gamma_c) * (v_max - v_min) + v_min
    image = image
    return image, label

def contrastaug(image, label):
    contrast_range = [0.9, 1.1]
    preserve = True
    factor = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = image.mean()
    if preserve:
        minm = image.min()
        maxm = image.max()
    image = (image - mean) * factor + mean
    if preserve:
        image[image < minm] = minm
        image[image > maxm] = maxm
    return image, label

def random_equalize_hist(image):
    image = skimage.exposure.equalize_hist(image)
    return image

def random_sharpening(image):
    blurred = ndimage.gaussian_filter(image, 3)
    blurred_filter = ndimage.gaussian_filter(blurred, 1)
    alpha = random.randrange(1, 10)
    image = blurred + alpha * (blurred - blurred_filter)
    return image

def min_max_normalize(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

class RandomGenerator(object):
    def __init__(self, output_size, SpatialAug=True, IntensityAug=True, NonlinearAug=False):
        self.output_size = output_size
        self.SpatialAug = SpatialAug
        self.IntensityAug = IntensityAug
        self.NonlinearAug = NonlinearAug
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if self.NonlinearAug:
            if random.random() > 0.5:
                image = nonlinear_transformation(image)
        if self.SpatialAug:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        if self.IntensityAug:
            if random.random() > 0.7:
                image, label = gammacorrection(image, label)
            if random.random() > 0.7:
                image, label = contrastaug(image, label)
            if random.random() > 0.7:
                image = random_equalize_hist(image)
            if random.random() > 0.7:
                image = random_sharpening(image)
            if random.random() > 0.7:
                image, label = gaussian_blur(image, label)
            if random.random() > 0.5:
                image, label = gaussian_noise(image, label)

        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
# [5]  Zhou Z, Qi L, Yang X, et al. Generalizable cross-modality medical image segmentation via style augmentation and dual normalization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 20856-20865.

import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb
"""  
this is for none linear transformation


"""

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=1000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def visualize_sample(sample):
    image = sample["image"][0].numpy()  # Assuming the batch size is 1
    label = sample["label"][0].numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image[0], cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    if len(label.shape) == 2:
        plt.imshow(label, cmap="gray")
    elif len(label.shape) == 3:
        plt.imshow(label[0], cmap="gray")  # Assuming label is 3D (batch_size, height, width)
    else:
        raise ValueError("Invalid shape for label data")

    plt.title("Original Label")

    plt.show()


if __name__ == "__main__":
    root_dir = r"F:\SFDA\data\data_preprocessed\SAML\3T"
    dataset = h5DataSet(base_dir=root_dir, split="train")
    db_train = h5DataSet(base_dir=root_dir, split="train",transform=transforms.Compose([
     RandomGenerator(output_size=(384, 384)),
    ]))
    train_loader = torch.utils.data.DataLoader(db_train, batch_size=24, shuffle=True, num_workers=1)
    for sample in train_loader:
        visualize_sample(sample)

    db_val = h5DataSet(base_dir=root_dir, split="val")

    valloader = torch.utils.data.DataLoader(db_val, batch_size=1, shuffle=False,
                       num_workers=1)
    for sample in train_loader:
        image = sample['image']
        label = sample['label']
        print(image.shape, label.shape)
        print(image.min(), image.max(), label.max())
