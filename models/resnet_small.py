'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
resnet same as the origin paper
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResSmall(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResSmall20():
    return ResSmall(BasicBlock, [3, 3, 3])


def ResSmall32():
    return ResSmall(BasicBlock, [5, 5, 5])


def ResSmall44():
    return ResSmall(Bottleneck, [7, 7, 7])


def ResSmall56():
    return ResSmall(Bottleneck, [9, 9, 9])


def ResSmall110():
    return ResSmall(Bottleneck, [18, 18, 18])


class BaseBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion * planes)
        #     )

    def forward(self, x):
        # print('x size', x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        return out


class RfSmall(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(RfSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.num_blocks = num_blocks
        self.layer_list = nn.ModuleList()
        self.shortcut_list = nn.ModuleList()
        layer1, shortcut1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2, shortcut2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3, shortcut3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer_list.extend([layer1, layer2, layer3])
        self.shortcut_list.extend([shortcut1, shortcut2, shortcut3])
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        shortcuts = nn.ModuleList()
        for stride in strides:
            # 64*64, 64*64 ..
            # 64*128(stride=2) 128*128 ..
            # 128*256(stride=2),256*256,..
            # 256*512(stride=2) 512*512 ..
            layers.append(block(self.in_planes, planes, stride))
            shortcut = nn.Sequential()
            if stride != 1 or self.in_planes != block.expansion * planes:
                shortcut = nn.Sequential(
                    nn.Conv2d(self.in_planes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion * planes)
                )
            shortcuts.append(shortcut)
            self.in_planes = planes * block.expansion
        return layers, shortcuts

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(4):
            for j in range(self.num_blocks[i]):
                # out = F.relu(self.bn1(self.conv1(x)))
                # out = self.bn2(self.conv2(out))
                # out += self.shortcut(x)
                # out = F.relu(out)
                # return out
                layer_i = self.layer_list[i]
                shortcut_i = self.shortcut_list[i]
                res = shortcut_i[j](out)
                out = layer_i[j](out)
                out += res
                out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RfSmall56(args):
    return RfSmall(block=BaseBlock, num_blocks=[9, 9, 9], args=args)


def RfSmall110(args):
    return RfSmall(block=BaseBlock, num_blocks=[18, 18, 18], args=args)


class LmRnnSmall(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(LmRnnSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.num_blocks = num_blocks
        self.layer_list = nn.ModuleList()
        self.shortcut_list = nn.ModuleList()
        layer1, shortcut1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2, shortcut2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3, shortcut3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer_list.extend([layer1, layer2, layer3])
        self.shortcut_list.extend([shortcut1, shortcut2, shortcut3])
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        shortcuts = nn.ModuleList()
        for stride in strides:
            # 64*64, 64*64 ..
            # 64*128(stride=2) 128*128 ..
            # 128*256(stride=2),256*256,..
            # 256*512(stride=2) 512*512 ..
            layers.append(block(self.in_planes, planes, stride))
            shortcut = nn.Sequential()
            if stride != 1 or self.in_planes != block.expansion * planes:
                shortcut = nn.Sequential(
                    nn.Conv2d(self.in_planes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion * planes)
                )
            shortcuts.append(shortcut)
            self.in_planes = planes * block.expansion
        return layers, shortcuts

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(4):
            for j in range(self.num_blocks[i]):
                # out = F.relu(self.bn1(self.conv1(x)))
                # out = self.bn2(self.conv2(out))
                # out += self.shortcut(x)
                # out = F.relu(out)
                # return out
                layer_i = self.layer_list[i]
                shortcut_i = self.shortcut_list[i]
                res = shortcut_i[j](out)
                out = layer_i[j](out)
                out += res
                out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def LmRnnSmall56(args):
    return LmRnnSmall(block=BaseBlock, num_blocks=[9, 9, 9], args=args)


def LmRnnSmall110(args):
    return LmRnnSmall(block=BaseBlock, num_blocks=[18, 18, 18], args=args)


def test():
    net = ResSmall20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
