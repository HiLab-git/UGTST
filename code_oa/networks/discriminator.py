import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

class MaskDiscriminator_SN(nn.Module):
    def __init__(self, num_classes=2, ndf=32):
        super(MaskDiscriminator_SN, self).__init__()
        self.conv0 = spectral_norm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
        self.conv1 = spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1))
        self.classifier = spectral_norm(nn.Linear(ndf*16, 1))
        self.avgpool = nn.AvgPool2d((7, 7))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.tanh = nn.Tanh()

    def forward(self, map):
        x = self.conv0(map)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.tanh(x)

        return x

class MaskDiscriminator(nn.Module):
    def __init__(self, num_classes=2, ndf=16):
        super(MaskDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Linear(ndf*16, 1)
        self.avgpool = nn.AvgPool2d((7, 7))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, map):
        x = self.conv0(map)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.sigmoid(x)

        return x

def create_circle_tensor(size):
    y, x = torch.meshgrid([torch.arange(size), torch.arange(size)])
    distance = torch.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
    circle_tensor = (distance <= size // 4).float().unsqueeze(0)
    return circle_tensor

def one_hot_encoder(n_classes, input_tensor):
    tensor_list = []

    for i in range(n_classes):
        temp_prob = (input_tensor == i).unsqueeze(1)
        tensor_list.append(temp_prob)

    output_tensor = torch.cat(tensor_list, dim=1)

    return output_tensor.float()
if __name__ == "__main__":
    # 示例
    model = MaskDiscriminator()
    print(model)
    map_input = torch.randn(1, 2, 384, 384)
    circle_tensor = create_circle_tensor(size=384)
    circle_tensor = one_hot_encoder(n_classes=2, input_tensor=circle_tensor)
    print(circle_tensor.shape)
    save_mode_path = (r'F:\SFDA\model\model_D\3T_al_labeled\Discriminator_best_model.pth')
    model.load_state_dict(torch.load(save_mode_path))
    output = model(circle_tensor)
    # output = model(map_input)
    print("output:", output)
