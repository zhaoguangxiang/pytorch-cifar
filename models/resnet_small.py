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
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResSmall20():
    return ResSmall(BasicBlock, [3, 3, 3])


def ResSmall32():
    return ResSmall(BasicBlock, [5, 5, 5])


def ResSmall44():
    return ResSmall(BasicBlock, [7, 7, 7])


def ResSmall56():
    return ResSmall(BasicBlock, [9, 9, 9])


def ResSmall110():
    return ResSmall(BasicBlock, [18, 18, 18])


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
        self.num_big_block = len(num_blocks)
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
        for i in range(self.num_big_block):
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
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # print('out size',out.size())
        out = self.linear(out)
        return out


def RfSmall56(args):
    return RfSmall(block=BaseBlock, num_blocks=[9, 9, 9], args=args)


def RfSmall110(args):
    return RfSmall(block=BaseBlock, num_blocks=[18, 18, 18], args=args)


class LmRnnSmall(nn.Module):
    # 只考虑分别设计三个rnn，然后bsz包含height 和width 的情况,层间使用残差连接。dim_type=channel,pass_hidden=0
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(LmRnnSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.num_blocks = num_blocks
        self.num_big_block = len(num_blocks)
        self.layer_list = nn.ModuleList()
        self.shortcut_list = nn.ModuleList()
        self.args = args
        self.rnn_list = nn.ModuleList()
        self.memory_type = args.memory_type
        # self.pass_hidden = args.pass_hidden
        self.rnn_ratio = args.rnn_ratio
        # self.dim_type = args.dim_type
        self.m_out_list =nn.ModuleList()
        self.rnn_memory_size_list = []
        layer1, shortcut1, rnn1, m_out_linear1, rnn_memory_size1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2, shortcut2, rnn2, m_out_linear2, rnn_memory_size2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3, shortcut3, rnn3, m_out_linear3, rnn_memory_size3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer_list.extend([layer1, layer2, layer3])
        self.shortcut_list.extend([shortcut1, shortcut2, shortcut3])
        self.rnn_list.extend([rnn1, rnn2, rnn3])
        self.m_out_list.extennd([m_out_linear1, m_out_linear2, m_out_linear3])
        self.rnn_memory_size_list.extend([rnn_memory_size1,rnn_memory_size2,rnn_memory_size3])
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        shortcuts = nn.ModuleList()
        if self.dim_type =='hw':
            rnn_input_size = block.expansion * planes
            rnn_memory_size = self.args.rnn_ratio * block.expansion * planes
        else:
            rnn_input_size = block.expansion * planes
            rnn_memory_size = self.args.rnn_ratio * block.expansion * planes
        if self.memory_type == 'rnn':
            rnn = torch.nn.RNNCell(rnn_input_size, rnn_memory_size, bias=True, nonlinearity='tanh')
        elif self.memory_type == 'lstm':
            rnn = torch.nn.LSTMCell(rnn_input_size, rnn_memory_size, bias=True)
        elif self.memory_type == 'gru':
            rnn = torch.nn.GRUCell(rnn_input_size, rnn_memory_size, bias=True)
        else:
            rnn = None
        if self.rnn_ratio != 1:
            m_out_linear = nn.Linear(self.rnn_memory_size, rnn_input_size)
        else:
            m_out_linear = None
        for i in range(num_blocks):
            # 对rnn来说，第一个残差连接虽然等维度，考虑到其他都是传h0，我就把当做和其他大块间残差的一样的
            stride = strides[i]
            # 16*16, 16*16 ..
            # 16*32(stride=2) 32*32 ..
            # 32*64(stride=2),64*64 ..
            layers.append(block(self.in_planes, planes, stride))
            if i == 0:
                shortcut = nn.Sequential()
                if stride != 1 or self.in_planes != block.expansion * planes:
                    shortcut = nn.Sequential(
                        nn.Conv2d(self.in_planes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(block.expansion * planes)
                    )
                shortcuts.append(shortcut)
            self.in_planes = planes * block.expansion
        return layers, shortcuts, rnn, m_out_linear,rnn_memory_size

    def set_m_rnn(self, x, rnn_memory_size):
        origin_bsz, height, width, _ = x.size()
        bsz = height * width * origin_bsz
        if self.memory_type in ['rnn',  'gru']:
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return hx
        if self.memory_type == 'lstm':
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                cx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return (hx, cx)

    def m_rnn(self, x,rnn_hidden):
        if self.memory_type in ['rnn', 'gru']:
            hx = self.rnn(x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = hx
        elif self.memory_type == 'lstm':
            hx, cx = self.rnn(x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden =(hx, cx)
        return m_output, rnn_hidden

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(self.num_big_block):
            if self.memory_type in ['rnn', 'gru']:
                rnn_hidden = self.set_m_rnn(out, self.rnn_memory_size_list[i])
            elif self.memory_type == 'lstm':
                rnn_hidden = self.set_m_rnn(out, self.rnn_memory_size_list[i])
            for j in range(self.num_blocks[i]):
                layer_i = self.layer_list[i]
                shortcut_i = self.shortcut_list[i]
                if j == 0:
                    res = shortcut_i[j](out)
                else:
                    m_out, rnn_hidden = self.m_rnn(out, rnn_hidden)
                    if self.m_out_list[i] is not None:
                        m_out = self.m_out_list[i](m_out)
                    res = m_out
                out = layer_i[j](out)
                out += res
                out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def LmRnnSmall56(args):
    return LmRnnSmall(block=BaseBlock, num_blocks=[9, 9, 9], args=args)


def LmRnnSmall110(args):
    return LmRnnSmall(block=BaseBlock, num_blocks=[18, 18, 18], args=args)

# class LmRnPhSmall():
#     def __init__(self):
#         shortcut = nn.Sequential()
#         if stride != 1 or self.in_planes != block.expansion * planes:
#             shortcut = nn.Sequential(
#                 nn.Conv2d(self.rnn_ratio * self.in_planes, self.rnn_ratio * block.expansion * planes, kernel_size=1,
#                           stride=stride,
#                           bias=False),
#                 nn.BatchNorm2d(block.expansion * planes)
#             )
#         shortcuts.append(shortcut)
def test():
    net = ResSmall20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
