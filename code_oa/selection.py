import argparse
import os
import random
import h5py
from tqdm import tqdm
from networks.unet import UNet
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from itertools import combinations 
import torch.nn as nn
import torch
import numpy as np
from tools import *
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/data_preprocessed/NPC_SMU/SMU', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='NPC/source_train', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--tta_num', type=int, default=8,
                    help='test time augmentation')
parser.add_argument('--spatial_aug', type=bool, default=False)
parser.add_argument('--intensity_aug', type=bool, default=True)
parser.add_argument('--C', type=int, default=4, help='Hyperparameter: capacity of Dtu')


def cluster_and_select_samples(combined_data, k=3):
    features = [data[2] for data in combined_data]
    X = np.array(features)
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(X, centroids)
    selected_samples = []
    for cluster_idx in range(k):
        cluster_samples = [combined_data[idx] for idx, cl_idx in enumerate(closest) if cl_idx == cluster_idx]
        closest_sample_idx = np.argmin([np.linalg.norm(sample[2] - centroids[cluster_idx]) for sample in cluster_samples])
        selected_samples.append(cluster_samples[closest_sample_idx])
    return selected_samples

def predict_with_tta_for_uncertainty_selection(image, net, output_path, parser, ratio=0.05):
    uncertainty = []
    feature_list = []
    img_name_list = []
    image_list = []
    pseudo_label = []
    label_list = []
    real_size = []
    size_esti = []
    for case in tqdm(image):
        h5f = h5py.File(parser.root_path + "/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        m = parser.tta_num
        image_copy = image
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, m, axis=0)
        for i in range(image.shape[0]):
            if parser.intensity_aug:
                image[i, :, :, :] = intensity_augmentor(image[i, :, :, :]).cpu().numpy()
        for ind in range(image.shape[1]):
            img_name = f'{case}_slice{ind}.h5'
            img_name_list.append(img_name)
            slice = image[:, ind, :, :]
            params = []
            input = torch.from_numpy(slice).unsqueeze(
                1).float().cpu()
            original_image = image_copy[ind, :, :]
            real_label = label[ind, :, :]
            pixel_count_real = np.count_nonzero(real_label)
            real_size.append(pixel_count_real)
            image_list.append(original_image)
            label_list.append(real_label)
            pixel_count_real = np.count_nonzero(real_label)
            real_size.append(pixel_count_real)
            volume_batch = input
            original_view = torch.from_numpy(original_image).unsqueeze(
                0).unsqueeze(0).float().cpu()
            spatial_augmentor = SpatialAugmentation(rotation_range=15)
            if parser.spatial_aug:
                for i in range(volume_batch.shape[0]):
                    volume_batch[i, :, :, :], params_i = spatial_augmentor.augment(volume_batch[i, :, :, :])
                    params.append(params_i)
            volume_batch = volume_batch.cuda()
            original_view = original_view.cuda()
            net.eval().cuda()
            with torch.no_grad():
                out_put, _ = net(volume_batch)
                out_main, features = net(original_view)
                features = features[-1]

                features_np = features.squeeze().cpu().numpy()
                
                features = features_np.flatten()
                feature_list.append(features)
                if parser.spatial_aug:
                    for i in range(out_put.shape[0]):
                        out_put[i, :, :, :] = spatial_augmentor.reverse_augment(out_put[i, :, :, :], params[i])
                output_prob = torch.softmax(out_main, dim=1)
                out_put = torch.softmax(out_put, dim=1)
                out_put = torch.mean(out_put, dim=0, keepdim=True)
                pse_label = torch.argmax(out_put, dim=1).squeeze()
                pse_label = pse_label.cpu().numpy()
                pseudo_label.append(pse_label)
                entropy = softmax_entropy(output_prob, softmax=False)
                entropy_np = entropy.cpu().detach().numpy()
                threshold = compute_entropy_density(entropy_np)
                selected_points = np.where(entropy_np > threshold)
                GAUA_uncertainty = np.mean(entropy_np[selected_points])
                uncertainty.append(GAUA_uncertainty)
                size_esti.append(len(selected_points[0]))

    budget = int(len(image_list) * ratio)
    print('The number of labeled slices is:', budget)
    combined_data = list(zip(img_name_list, uncertainty, feature_list))
    combined_data.sort(key=lambda x: x[1], reverse=True)
    uncertainty_selected_samples = combined_data[:budget * parser.C]
    method = 'UGTST'
    selected_samples = cluster_and_select_samples(uncertainty_selected_samples, k=budget)
    selected_img_names = [sample[0] for sample in selected_samples]

    all_data = list(zip(img_name_list, image_list, pseudo_label, label_list, uncertainty))
    for i, data in enumerate(all_data):
        img_name, image, pseudo_label, true_label, uncertainty = data
        if img_name in selected_img_names:
            all_data[i] = (img_name, image, true_label, -1)
        else:
            all_data[i] = (img_name, image, pseudo_label, uncertainty)
    all_data = [(img_name, image, pseudo_label, uncertainty) for img_name, image, pseudo_label, uncertainty in all_data]
    all_data.sort(key=lambda x: x[3])

    Dts_percent = int(len(all_data) * (1 - ratio * (parser.C - 1)))
    Dts_img_names = [data[0] for data in all_data[:Dts_percent]]
    with open(f"{output_path}/stage1_slice_{method}.txt", "w") as f:
        for name in Dts_img_names:
            f.write(name + '\n')

    all_img_names = [data[0] for data in all_data]
    with open(f"{output_path}/all_slice_{method}.txt", "w") as f:
        for name in all_img_names:
            f.write(name + '\n')
    os.makedirs(os.path.join(output_path, 'slice_pseudo'), exist_ok=True)
    for img_name, image, label, jsd in all_data:
        hdf5_file_path = os.path.join(output_path,
                                      f'slices_pseudo/{img_name}')
        with h5py.File(hdf5_file_path, "w") as f:
            f.create_dataset('image', data=image, compression="gzip")
            f.create_dataset('label', data=label, compression="gzip")
    with open(os.path.join(output_path, f'selection_{method}.txt'), 'w') as f:
        for item in selected_samples:
            f.write(f"{item[0]}: {item[1]}\n")



if __name__ == '__main__':
    parser = parser.parse_args()
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    torch.cuda.manual_seed(parser.seed)
    with open(parser.root_path + f'/trainlist.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/".format(parser.exp)
    net = UNet(in_chns=1, class_num=parser.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'UNet_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    predict_with_tta_for_uncertainty_selection(image_list, net, output_path=parser.root_path, parser=parser)
