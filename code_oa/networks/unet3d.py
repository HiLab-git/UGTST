
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet

model = PlainConvUNet(
    input_channels=1, 
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2],
    num_classes=2,
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-05, "affine": True},
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={"inplace": True}
)

print(model)
