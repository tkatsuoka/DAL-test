import numpy as np
import torch
from scipy.stats import norm, t, skewnorm, gennorm, exponnorm
from scipy.integrate import quad
from scipy.optimize import brentq
from torch.utils.data import Dataset


def standardize(rv, param):
    if type(rv) == str:
        rv = get_rv_from_name(rv)
    mean = rv.mean(param)
    std = rv.std(param)
    return rv(param, loc=-mean / std, scale=1 / std)


def wasserstein_distance(rv):
    def func(x):
        return np.abs(rv.cdf(x) - norm.cdf(x))

    return quad(func, -np.inf, np.inf)[0]


def target_function(rv, target):
    def func(param):
        return wasserstein_distance(standardize(rv, param)) - target

    return func


def get_rv_from_name(rv_name):
    rv_dict = {
        "skewnorm": skewnorm,
        "exponnorm": exponnorm,
        "gennormsteep": gennorm,
        "t": t,
    }
    return rv_dict[rv_name]


def binary_search(rv_name, distance):
    rv = get_rv_from_name(rv_name)
    func = target_function(rv, distance)
    range_dict = {
        "skewnorm": (1e-4, 30.0),
        "exponnorm": (1e-4, 15.0),
        "gennormsteep": (1e-1, 2.0 - 1e-4),
        "t": (3.0, 200.0),
    }
    return brentq(func, *range_dict[rv_name])


def standardized_rv_at_distance(distribution, distance):
    params_dict = {
        "skewnorm": {
            "0.03": 1.141679535895037,
            "0.06": 1.668027646656356,
            "0.09": 2.253555993158534,
            "0.12": 3.052977442461724,
            "0.15": 4.441693019739707,
        },
        "exponnorm": {
            "0.03": 0.5274333543184184,
            "0.06": 0.7361945074942922,
            "0.09": 0.9307079975424131,
            "0.12": 1.1365153042836023,
            "0.15": 1.372114598160624,
        },
        "gennormsteep": {
            "0.03": 1.685486347382175,
            "0.06": 1.446878209856004,
            "0.09": 1.2592111500311147,
            "0.12": 1.1075283854228473,
            "0.15": 0.9822742249929434,
        },
        "t": {
            "0.03": 13.911718115376004,
            "0.06": 7.606345474941293,
            "0.09": 5.498186625845221,
            "0.12": 4.441398730633352,
            "0.15": 3.8067196925891835,
        },
    }
    if distance in [0.03, 0.06, 0.09, 0.12, 0.15]:
        param = params_dict[distribution][str(distance)]
    else:
        param = binary_search(distribution, distance)

    rv = get_rv_from_name(distribution)
    return standardize(rv, param)


def generate_images_iid(num, img_size, scale, signal=0, seed=0):
    rng = np.random.default_rng(seed=seed)
    images = []
    labels = []
    masks = []
    for _ in range(num):
        image = rng.normal(0, scale, (img_size, img_size))
        mask = np.zeros((img_size, img_size))
        abnormal_size = img_size // 3
        abnormal_x = rng.integers(0, img_size - abnormal_size)
        abnormal_y = rng.integers(0, img_size - abnormal_size)
        image[
            abnormal_x : abnormal_x + abnormal_size,
            abnormal_y : abnormal_y + abnormal_size,
        ] += signal
        label = 0
        if signal != 0:
            mask[
                abnormal_x : abnormal_x + abnormal_size,
                abnormal_y : abnormal_y + abnormal_size,
            ] = 1
            label = 1
        images.append(image)
        masks.append(mask)
        labels.append(label)
    images = np.array(images)
    masks = np.array(masks)
    labels = np.array(labels)
    images = images.reshape(num, 1, img_size, img_size)
    masks = masks.reshape(num, 1, img_size, img_size)
    return images, masks, labels


def generate_images_corr(num, img_size, signal, seed=0):
    rng = np.random.default_rng(seed=seed)
    cov = [[] for _ in range(img_size * img_size)]
    for i in range(img_size * img_size):
        for j in range(img_size * img_size):
            cov[i].append(np.abs(i - j))
    cov = np.array(cov)
    cov = np.power(0.5, cov)

    images = rng.multivariate_normal(np.zeros(img_size * img_size), cov, num)
    images = images.reshape(num, 1, img_size, img_size)
    for i in range(num):
        abnormal_size = img_size // 3
        abnormal_x = rng.integers(0, img_size - abnormal_size)
        abnormal_y = rng.integers(0, img_size - abnormal_size)
        images[
            i,
            0,
            abnormal_x : abnormal_x + abnormal_size,
            abnormal_y : abnormal_y + abnormal_size,
        ] += signal
    return images, cov


def generate_images_non_iid(image_size, distribution, distance, num_samples):
    images = []
    rvs = standardized_rv_at_distance(distribution=distribution, distance=distance)
    for _ in range(num_samples):
        images.append(rvs.rvs(size=(1, image_size, image_size)))
    images = np.array(images)
    images = images.reshape(num_samples, 1, image_size, image_size)
    return images


class SyntheticDataset(Dataset):
    """Training images of the diffusion models (iid Gaussian noise)."""

    def __init__(
        self,
        num_samples,
        img_size,
        scale=1,
        signal=0,
        seed=0,
    ):
        self.images, self.masks, self.labels = generate_images_iid(
            num_samples, img_size, scale, signal, seed
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float()
        label = torch.tensor([self.labels[idx]]).float()
        return image, mask, label
