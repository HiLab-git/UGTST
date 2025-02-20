import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.unet import UNet, UNet_DST, UNet_UPL
from scipy import ndimage
# from networks.efficientunet import UNet,
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/data_preprocessed/NPC_SMU/SMU', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='NPC_ablation/SMU_UGTST+_5%_tst_with_cons', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--largest_component', type=bool, default=True,
                    help='get the largest component')
parser.add_argument('--target_set', type=str, default='val',
                    help='target_set')

def get_largest_component(image):
    dim = len(image.shape)
    if(image.sum() == 0 ):
        # print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output

def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.assd(pred, gt, spacing)
    hd95 = metric.binary.hd95(pred, gt, spacing)
    return dice, asd, hd95


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    org_spacing = h5f['spacing'][:]

    spacing = [org_spacing[2], org_spacing[0], org_spacing[1]]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (320 / x, 320 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval().cuda()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            elif FLAGS.model == "UNet_UPL":
                outputs1, outputs2, outputs3, outputs4 = net(input)
                out_main = (outputs1 + outputs2 + outputs3 + outputs4) / 4.0
            else:
                out_main, _ = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 320, y / 320), order=0)
            prediction[ind] = pred
    if FLAGS.largest_component:
        prediction = get_largest_component(prediction)
    first_metric = calculate_metric_percase(prediction == 1, label == 1, spacing=spacing)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))

    img_itk.SetSpacing(org_spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(org_spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing(org_spacing)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric


def Inference(FLAGS):

    with open(FLAGS.root_path + f'/{FLAGS.target_set}list.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/".format(FLAGS.exp)
    pre = os.path.basename(FLAGS.root_path)
    test_save_path = "../model/{}/{}_{}_{}_predictions/".format(
        FLAGS.exp, FLAGS.model, pre, FLAGS.target_set)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    if FLAGS.model == 'UNet_UPL':
        net = UNet_UPL(in_chns=1, class_num=FLAGS.num_classes)
    elif FLAGS.model == 'UNet_DST':
        net = UNet_DST(in_chns=1, class_num=FLAGS.num_classes)
    else:
        net = UNet(in_chns=1, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, f'UNet_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    dice_scores = []
    assd_values = []
    hd95_values = []

    with open(os.path.join(test_save_path, 'metrics.txt'), 'w') as f:
        f.write("Case\tDice\tHD95\tASSD\n")
        for case in tqdm(image_list):
            metrics = test_single_volume(case, net, test_save_path, FLAGS)
            dice_scores.append(metrics[0])
            hd95_values.append(metrics[2])
            assd_values.append(metrics[1])
            f.write(f"{case}\t{metrics[0]}\t{metrics[2]}\t{metrics[1]}\n")

    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)

    avg_hd95 = np.mean(hd95_values)
    std_hd95 = np.std(hd95_values)

    avg_asd = np.mean(assd_values)
    std_asd = np.std(assd_values)

    avg_metrics = {
        'avg_dice': avg_dice,
        'std_dice': std_dice,
        'avg_hd95': avg_hd95,
        'std_hd95': std_hd95,
        'avg_assd': avg_asd,
        'std_assd': std_asd,
    }
    formatted_metrics = {
        'dice': f'{avg_metrics["avg_dice"] * 100:.2f}' + f'±{avg_metrics["std_dice"] * 100:.2f}',
        'hd95': f'{avg_metrics["avg_hd95"]:.2f}±{avg_metrics["std_hd95"]:.2f}',
        'assd': f'{avg_metrics["avg_assd"]:.2f}±{avg_metrics["std_assd"]:.2f}',
    }
    print("Formatted Metrics:")
    for key, value in formatted_metrics.items():
        print(f'{key}: {value}')

    with open(os.path.join(test_save_path, 'overall_metrics.txt'), 'w') as f:
        for key, value in avg_metrics.items():
            f.write(f'{key}: {value}\n')
        for key, value in formatted_metrics.items():
            f.write(f'{key}: {value}\n')

    return avg_metrics

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    # print((metric[0]+metric[1]+metric[2])/3)
