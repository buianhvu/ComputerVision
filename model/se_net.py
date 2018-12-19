from model.res_net import *
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_planes, r=4):
        super(SEBlock, self).__init__()
        self.in_planes = in_planes
        if self.in_planes % r != 0:
            raise ValueError('in_planes must be divisible with r (default=4)')
        self.temp_planes = int(self.in_planes/r)
        self.fc1 = nn.Linear(self.in_planes, self.temp_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.temp_planes, self.in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.reshape(out.size()[0], -1, 1, 1)
        out = out * x
        return out


class SENet(nn.Module):
    def __init__(self, layers: list, in_planes=3, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = in_planes
        self.layers = []
        for i, feature in enumerate(layers):
            self.layers.append(self.make_layers(feature[0], feature[1], feature[2]))
        self.se_layers = nn.Sequential(*self.layers)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.in_planes, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.initialize()

    def forward(self, x):
        x = self.se_layers(x)
        if x.size()[3] >=8:
            x = self.max_pool(x)
        x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, kernel_size, planes, layers_num, padding=1, stride=1):
        layers = []
        in_planes = self.in_planes
        for i in range(layers_num):
            conv = nn.Conv2d(self.in_planes, planes, kernel_size, padding=padding, stride=stride)
            self.in_planes = planes
            block = SEBlock(self.in_planes)
            bn = nn.BatchNorm2d(planes)
            relu = nn.ReLU(inplace=True)
            layers.append(conv)
            layers.append(block)
            layers.append(bn)
            layers.append(relu)

        return nn.Sequential(*layers)


def se_default(in_planes=3, num_classes=10):
    layers = [(3, 64, 2), (3, 128, 2), (3, 256, 4), (3, 512, 4), [3, 512, 4]]
    return SENet(layers, in_planes)



