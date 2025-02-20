import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
import re
from scipy import ndimage
image_pattern = re.compile(r'Case(\d+).nii.gz')

data_raw_path = "/media/disk2t_solid/zihao/SFDA/data/data_raw/Prostate"
data_preprocessed_path = "/media/disk2t_solid/zihao/SFDA/data/data_preprocessed/Prostate"

subfolders = glob.glob(os.path.join(data_raw_path, "*"))

for subfolder in subfolders:
    image_paths = glob.glob(os.path.join(subfolder, "*.nii.gz"))

    subfolder_name = os.path.basename(subfolder)

    for image_path in image_paths:

        image_match = image_pattern.search(os.path.basename(image_path))
        if image_match:
            case_number = image_match.group(1)
            print("Processing Case:", case_number)

            label_path = os.path.join(subfolder, f'Case{case_number}_segmentation.nii.gz')

            if os.path.exists(label_path):
                print("Processing Label:", label_path)

                img_itk = sitk.ReadImage(image_path)
                origin = img_itk.GetOrigin()
                spacing = img_itk.GetSpacing()
                direction = img_itk.GetDirection()
                image = sitk.GetArrayFromImage(img_itk)

                label_itk = sitk.ReadImage(label_path)
                label = sitk.GetArrayFromImage(label_itk).astype(np.uint8)
                image = (image - image.min()) / (image.max() - image.min())
                image = image.astype(np.float32)

                num_slices = image.shape[0]

                item = image_path.split("/")[-1].split(".")[0]
                if image.shape != label.shape:
                    print("Error")
                print(item)
                file_path = os.path.join(data_preprocessed_path, subfolder_name)
                hdf5_file_path = os.path.join(file_path,
                                              f'Case{case_number}.h5')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                    os.makedirs(os.path.join(file_path, 'slices'))
                with h5py.File(hdf5_file_path, 'w') as f:
                    f.create_dataset('image', data=image, compression="gzip")
                    f.create_dataset('label', data=label, compression="gzip")
                    f.create_dataset('spacing',data=spacing, compression="gzip")
                
                for slice_ind in range(num_slices):
                    hdf5_file_path = os.path.join(file_path,
                                                  f'slices/Case{case_number}_slice{slice_ind}.h5')

                    with h5py.File(hdf5_file_path, 'w') as f:
                        f.create_dataset('image', data=image[slice_ind], compression="gzip")
                        f.create_dataset('label', data=label[slice_ind], compression="gzip")
print("Preprocess completed.")
