from model.res_net import *


class SEBlock(nn.Module):
    def __init__(self, input_size, in_planes, r=4):
        super(SEBlock, self).__init__()
        self.in_planes = in_planes
        self.temp_planes = int(self.in_planes/r)
        self.global_avg_pool = nn.AvgPool2d(input_size)
        self.fc1 = nn.Linear(self.in_planes, self.temp_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.temp_planes, self.in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.reshape(self.in_planes, 1)
        out = out * x
        return out


class SENet(nn.Module):
    def __init__(self, layers: list, in_planes=3, input_size=32, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = in_planes
        self.input_size = input_size
        self.layers = []
        for i, feature in enumerate(layers):
            self.layers.append(self.make_layers(feature[0], feature[1], feature[2]))
        self.se_layers = nn.Sequential(*self.layers)
        self.fc1 = nn.Linear(self.in_planes, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.initialize()

    def forward(self, x):
        x = self.se_layers(x)
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
            block = SEBlock(self.input_size, self.in_planes)
            bn = nn.BatchNorm2d(planes)
            relu = nn.ReLU(inplace=True)
            layers.append(conv)
            layers.append(block)
            layers.append(bn)
            layers.append(relu)
        if in_planes == planes:
            max_pool = None
        else:
            if self.input_size < 5:
                max_pool = None
            elif in_planes != planes:
                max_pool = nn.MaxPool2d(2, 2)
            else:
                max_pool = None
            # elif self.input_size % 2 == 0:
            #     max_pool = nn.MaxPool2d(2, 2)
            #     self.input_size = self.input_size/2
            # else:
            #     max_pool = None
        if max_pool is not None:
            layers.append(max_pool)
        return nn.Sequential(*layers)


def se_default(in_planes=3, input_size=32):
    layers = [(3, 64, 2), (3, 128, 2), (3, 256, 4), (3, 512, 4), [3, 512, 4]]
    return SENet(layers, in_planes, input_size)



