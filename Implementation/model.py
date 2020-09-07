import torch
import torch.nn as nn
import torch.nn.functional as F

try: from torch.hub import load_state_dict_from_url
except ImportError: from torch.utils.model_zoo import load_url as load_state_dict_from_url

torch.manual_seed(0)


# Model
# - ImageNet_resnext50_32x4d, ImageNet_resnext101_32x8d
# - CIFAR_resnext29_16x4d

# Pretrained model weights url (pretrained on ImageNet)
pretrained_model_urls = {
    'resnext50': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'
}


# +
class Bottleneck_ImageNet(nn.Module):
    expansion = 4
    def __init__(self, inplanes, stride=1, cardinality=None, first=False):
        super(Bottleneck_ImageNet, self).__init__()
        self.stride = stride
        self.first = first
        
        if self.stride != 1:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=self.stride, bias=False)
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, 2*inplanes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(2*inplanes)
            )
        elif first == True:
            self.conv1 = nn.Conv2d(inplanes//2, inplanes, kernel_size=1, stride=self.stride, bias=False)
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes//2, 2*inplanes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(2*inplanes)
            )
        else:
            self.conv1 = nn.Conv2d(2*inplanes, inplanes, kernel_size=1, stride=self.stride, bias=False)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False, groups=cardinality[0])
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, 2*inplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2*inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if (self.stride == 1) and (self.first == False):
            identity = x
        else:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
    
    
class Bottleneck_CIFAR(nn.Module):
    expansion = 4
    def __init__(self, inplanes, stride=1, cardinality=None, first=False):
        super(Bottleneck_CIFAR, self).__init__()
        self.stride = stride
        self.first = first
        
        if self.stride != 1:
            self.conv1 = nn.Conv2d(4*inplanes, inplanes, kernel_size=1, stride=self.stride, bias=False)
            self.downsample = nn.Sequential(
                nn.Conv2d(4*inplanes, 4*inplanes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(4*inplanes)
            )
        elif first == True:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=self.stride, bias=False)
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, 4*inplanes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(4*inplanes)
            )
        else:
            self.conv1 = nn.Conv2d(4*inplanes, inplanes, kernel_size=1, stride=self.stride, bias=False)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False, groups=cardinality[0])
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, 4*inplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if (self.stride == 1) and (self.first == False):
            identity = x
        else:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# -

class _resnext(nn.Module):

    def __init__(self, mode, block, cardinality, layers, num_classes=1000):
        super(_resnext, self).__init__()
        self.mode = mode
        self.cardinality = cardinality
        
        if self.mode == 'ImageNet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 128, layers[0])
            self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(block.expansion*512, num_classes)
            
        if self.mode == 'CIFAR':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(block.expansion*64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(planes, stride=stride, cardinality=self.cardinality, first=True)]
        for _ in range(1, blocks):
            layers.append(block(planes, cardinality=self.cardinality))

        return nn.Sequential(*layers)

    def _forward_ImageNet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _forward_CIFAR(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        if self.mode == 'ImageNet':
            return self._forward_ImageNet(x)
        elif self.mode == 'CIFAR':
            return self._forward_CIFAR(x)


# Model info
cfgs = {
    # ImageNet Model
    50 : ['ImageNet', Bottleneck_ImageNet, [32, 4], [3, 4, 6, 3]],
    101 : ['ImageNet', Bottleneck_ImageNet, [32, 8], [3, 4, 23, 3]],
    
    # CIFAR Model
    29 : ['CIFAR', Bottleneck_CIFAR, [16, 4], [3, 3, 3]]
}


def resnext(depth, num_classes, pretrained):
    
    model = _resnext(mode=cfgs[depth][0], block=cfgs[depth][1], cardinality=cfgs[depth][2], layers=cfgs[depth][3], num_classes=num_classes)
    arch = 'resnext'+str(depth)
    
    if pretrained and (num_classes == 1000) and (arch in pretrained_model_urls):
        state_dict = load_state_dict_from_url(pretrained_model_urls[arch], progress=True)
        model.load_state_dict(state_dict)
    elif pretrained:
        raise ValueError('No pretrained model in resnext {} model with class number {}'.format(depth, num_classes))
            
    return model
