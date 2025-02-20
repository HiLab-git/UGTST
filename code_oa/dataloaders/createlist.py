import os
import random
import re
data_folder = "/media/disk2t_solid/zihao/SFDA/data/data_preprocessed/3D_pancreas/MCF"

file_list = [file for file in os.listdir(data_folder) if file.endswith(".h5")]

random.shuffle(file_list)

split_point1 = int(0.7 * len(file_list))
split_point2 = int(0.8 * len(file_list))

train_set = file_list[:split_point1]
val_set = file_list[split_point1:split_point2]
test_set = file_list[split_point2:]

with open(os.path.join(data_folder, "trainlist.txt"), 'w') as train_file:
    train_file.write('\n'.join(train_set))

with open(os.path.join(data_folder, "vallist.txt"), 'w') as val_file:
    val_file.write('\n'.join(val_set))

with open(os.path.join(data_folder, "testlist.txt"), 'w') as test_file:
    test_file.write('\n'.join(test_set))

trainlist_path = os.path.join(data_folder, 'trainlist.txt')
with open(trainlist_path, 'r') as f:
    file_names = f.read().splitlines()

slice_file_names = []
slices_folder_path = os.path.join(data_folder,'slices')
print(slices_folder_path)

for file_name in file_names:
    case_name = file_name.split('_')[0]
    case_name = case_name.replace('.h5','')
    pattern = re.compile(f'{case_name}_slice(\d+).h5')
    print(case_name)
    slice_files = [f for f in os.listdir(slices_folder_path) if re.match(pattern, f)]
    slice_file_names.extend(slice_files)

output_path = os.path.join(data_folder,'train_slices.txt')
with open(output_path, 'w') as f:
    for slice_file_name in slice_file_names:
        f.write(slice_file_name + '\n')

print("Finished!")