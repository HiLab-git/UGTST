import matplotlib.pyplot as plt
import torchvision.transforms.functional
import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.signal import find_peaks

def embedding(feature, mask):
    last_feature = feature[-1]
    mask = torch.nn.AdaptiveAvgPool2d(mask, output_size=(128, 128)).argmax(1)
    embedding = torch.nn.MaxPool2d()


def gaussian_noise(image):
    mean = 0
    std = 0.05
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    return image


def gaussian_blur(image):
    std_range = [0, 1]
    std = np.random.uniform(std_range[0], std_range[1])
    image = gaussian_filter(image, std, order=0)
    return image


def clip_filter(image, lower_percentile=0.5, upper_percentile=99.5):
    lower_limit = np.percentile(image, lower_percentile)
    upper_limit = np.percentile(image, upper_percentile)

    clipped_image = np.clip(image, lower_limit, upper_limit)
    v_min = clipped_image.min()
    v_max = clipped_image.max()
    clipped_image = (clipped_image - v_min) / (v_max - v_min)
    return clipped_image


def gammacorrection(image):
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
    return image


def contrastaug(image):
    contrast_range = [0.8, 1.2]
    preserve = True
    factor = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = image.mean()
    if preserve:
        minm = image.min()
        maxm = image.max()
    image = (image - mean) * factor + mean
    image[image < minm] = minm
    image[image > maxm] = maxm

    return image


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


# bernstein_poly(i, n, t)：计算伯恩斯坦多项式，其中 i 为多项式的次数，n 为多项式的阶数，t 为参数化值。该函数用于计算贝塞尔曲线中的权重系数。
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

def intensity_augmentor(sample):
    # Check if input is a tensor, if so, convert to NumPy (on CPU) for augmentation
    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()  # Move tensor to CPU and convert to NumPy

    image = sample
    if random.random() > 0.5:
        image = nonlinear_transformation(image)
    if random.random() > 0.5:
        image = contrastaug(image)
    if random.random() > 0.5:
        image = clip_filter(image)
    if random.random() > 0.5:
        image = gammacorrection(image)
    if random.random() > 0.5:
        image = gaussian_blur(image)
    if random.random() > 0.5:
        image = gaussian_noise(image)

    # After augmentation, convert back to Tensor and move to CUDA if necessary
    sample = torch.from_numpy(image).float()  # Convert back to Tensor
    if torch.cuda.is_available():
        sample = sample.cuda()  # Move Tensor back to CUDA if needed

    return sample


def softmax_confidence(preds, softmax):
    if softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
    CONF = torch.max(preds, 1)[0]
    CONF *= -1  # The small the better --> Reverse it makes it the large the better
    return CONF

def softmax_entropy(preds, softmax=True):
    # Softmax Entropy
    if softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
    ENT = torch.sum(-preds * torch.log2(preds + 1e-12), dim=1)  # The large the better
    return ENT


class SpatialAugmentation:
    def __init__(self, rotation_range=0):
        self.rotation_range = rotation_range

    def augment(self, img):
        rotation_90 = int(np.random.choice([0]))
        flip = int(np.random.choice([2, 3]))
        rotation_angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        # img = torch.flip(img, [flip])
        img = torchvision.transforms.functional.rotate(img, rotation_90)
        # img = torchvision.transforms.functional.rotate(img, rotation_angle)

        return img, {'rotation_90': rotation_90, 'rotation_angle': rotation_angle}

    def reverse_augment(self, augmented_img, params):
        rotation_90 = -params['rotation_90']
        rotation_angle = -params['rotation_angle']
        # flip = params['flip']
        # reversed_img = torchvision.transforms.functional.rotate(augmented_img, rotation_angle)
        reversed_img = torchvision.transforms.functional.rotate(augmented_img, rotation_90)
        # reversed_img = torch.flip(reversed_img, [flip])

        return reversed_img


def compute_entropy_density(entropy_np):
    entropy_flat = entropy_np.flatten()
    hist, bins = np.histogram(entropy_flat, bins=100, density=True)
    peaks, _ = find_peaks(hist)
    threshold = bins[peaks[0]]
    return threshold



def cluster_and_select_samples(combined_data, k=3):

    features = [data[2] for data in combined_data]
    X = np.array(features)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(X, centroids)

    selected_samples = []

    for cluster_idx in range(k):
        cluster_samples = [combined_data[idx] for idx, cl_idx in enumerate(closest) if cl_idx == cluster_idx]
        closest_sample_idx = np.argmin([np.linalg.norm(sample[2] - centroids[cluster_idx]) for sample in cluster_samples])
        selected_samples.append(cluster_samples[closest_sample_idx])

    return selected_samples


def js_divergence(p, q):
    p = p.cpu().numpy()
    q = q.cpu().numpy()

    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def create_circle_tensor(size):
    y, x = torch.meshgrid([torch.arange(size), torch.arange(size)])
    distance = torch.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
    circle_tensor = (distance <= size // 4).float().unsqueeze(0).unsqueeze(0)
    return circle_tensor


def visualize_tensor(tensor):
    plt.imshow(tensor[0, 0, :, :], cmap='gray')
    plt.show()


def one_hot_encoder(n_classes, input_tensor):
    tensor_list = []

    for i in range(n_classes):
        temp_prob = (input_tensor == i).unsqueeze(1)
        tensor_list.append(temp_prob)

    output_tensor = torch.cat(tensor_list, dim=1)

    return output_tensor.float()


def smooth_segmentation_labels(n_classes, target_tensor):

    tensor_list = []
    for i in range(n_classes):
        class_mask = (target_tensor == i).unsqueeze(1)
        temp_prob = class_mask.float()
        smoothing_factor = torch.rand_like(temp_prob) * 0.1
        smoothing = (1.0 - smoothing_factor) * temp_prob + smoothing_factor / n_classes
        tensor_list.append(smoothing)

    output_tensor = torch.cat(tensor_list, dim=1)

    return output_tensor.float()

def add_noise_boxes(incoming_mask, n_classes, image_size, mask_type, n_boxes=3, probability=None, real_mask=False):
    if probability is None:
        probability = {'random': 1.0, 'jigsaw': 1.0, 'zeros': 1.0}
    for p in probability.values():
        assert 0.0 <= p <= 1.0

    if type(mask_type) is not list:
        assert type(mask_type) == str
        mask_type = [mask_type]

    def _py_corrupt(mask):
        mask = mask.numpy()
        mask = mask.astype(np.float32)  # Ensure the array is in float32
        jigsaw_op = np.random.choice([True, False], p=[probability['jigsaw'], 1.0 - probability['jigsaw']])
        zeros_op = np.random.choice([True, False], p=[probability['zeros'], 1.0 - probability['zeros']])
        random_op = np.random.choice([True, False], p=[probability['random'], 1.0 - probability['random']])
        if not (jigsaw_op or zeros_op):
            random_op = True

        for _ in range(n_boxes):

            def get_box_params(low, high):
                r = np.random.randint(low=low, high=high)
                mcx = np.random.randint(r + 1, image_size[0] - r - 1)
                mcy = np.random.randint(r + 1, image_size[1] - r - 1)
                return r, mcx, mcy

            if 'random' in mask_type and random_op:
                r, mcx, mcy = get_box_params(low=1, high=5)
                mask[:, mcx - r:mcx + r, mcy - r:mcy + r] = 0
                mask[:, mcx - r:mcx + r, mcy - r:mcy + r] = 1
            if 'jigsaw' in mask_type and jigsaw_op:
                ll = np.min([image_size[0], image_size[1]]) // 10
                hh = np.min([image_size[0], image_size[1]]) // 5
                r, mcx, mcy = get_box_params(low=ll, high=hh)
                mask[mcx - r:mcx + r, mcy - r:mcy + r] = 0
                mcx_src = np.random.randint(r + 1, image_size[0] - r - 1)
                mcy_src = np.random.randint(r + 1, image_size[1] - r - 1)
                mask_copy = mask.copy()
                mask[:, mcx - r:mcx + r, mcy - r:mcy + r] = mask_copy[:, mcx_src - r:mcx_src + r,
                                                            mcy_src - r:mcy_src + r]
            if 'zeros' in mask_type and zeros_op:
                r, mcx, mcy = get_box_params(low=1, high=10)
                mask[:, mcx - r:mcx + r, mcy - r:mcy + r] = 0
                mask[:, mcx - r:mcx + r, mcy - r:mcy + r] = 1
        return mask

    incoming_mask = incoming_mask.cpu()
    if real_mask:
        incoming_mask = one_hot_encoder(n_classes=n_classes, input_tensor=incoming_mask)
    noisy_masks = [_py_corrupt(m) for m in incoming_mask]

    mask = torch.from_numpy(np.array(noisy_masks))
    mask = mask.cuda()
    return mask


# binary_tensor = torch.randint(2, size=(1, 384, 384), dtype=torch.uint8)
# predict_tensor = torch.randn(1, 2, 384, 384)
# noised_tensor = add_noise_boxes(binary_tensor, 2, image_size=[384, 384], mask_type=['jigsaw', 'random', 'zeros'],
#                                 real_mask=True)
# noised_pre_tensor = add_noise_boxes(predict_tensor, 2, image_size=[384, 384], mask_type=['jigsaw', 'random', 'zeros'])
# circle_tensor = create_circle_tensor(size=384)
# circle_noise_tensor = add_noise_boxes(circle_tensor, 2, image_size=[384, 384], n_boxes=20,
#                                       probability={'random': 0.9, 'jigsaw': 0.5, 'zeros': 0.5}, mask_type=['random'])
# print('noised shape:', noised_tensor.shape)
# visualize_tensor(circle_tensor)
# visualize_tensor(circle_noise_tensor)
# print('noised shape:', noised_tensor.shape)
# print('noised pre shape:', noised_pre_tensor.shape)
