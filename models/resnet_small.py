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
import numpy as np


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
        self.rnn_list = nn.ModuleList()
        self.m_out_list = nn.ModuleList()
        self.rnn_memory_size_list = []

        self.args = args
        self.memory_type = args.memory_type
        # self.pass_hidden = args.pass_hidden
        self.rnn_ratio = args.rnn_ratio
        # self.dim_type = args.dim_type
        self.rnn_res = args.rnn_res

        layer1, shortcut1, rnn1, m_out_linear1, rnn_memory_size1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2, shortcut2, rnn2, m_out_linear2, rnn_memory_size2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3, shortcut3, rnn3, m_out_linear3, rnn_memory_size3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.layer_list.extend([layer1, layer2, layer3])
        self.shortcut_list.extend([shortcut1, shortcut2, shortcut3])
        self.rnn_list.extend([rnn1, rnn2, rnn3])
        self.m_out_list.extend([m_out_linear1, m_out_linear2, m_out_linear3])
        self.rnn_memory_size_list.extend([rnn_memory_size1,rnn_memory_size2,rnn_memory_size3])

        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        shortcuts = nn.ModuleList()
        rnn_input_size = block.expansion * planes
        rnn_memory_size = int(self.args.rnn_ratio * block.expansion * planes)
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
        origin_bsz, channel, height, width, = x.size()
        bsz = height * width * origin_bsz
        if self.memory_type in ['rnn',  'gru']:
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return hx
        if self.memory_type == 'lstm':
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                cx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return (hx, cx)

    def m_rnn(self, x, rnn, rnn_hidden):
        origin_bsz, channel, height, width  = x.size()
        in_x = x.permute(0, 2, 3, 1).reshape(origin_bsz*height*width, channel)
        if self.memory_type in ['rnn', 'gru']:
            hx = rnn(in_x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = hx
        elif self.memory_type == 'lstm':
            hx, cx = rnn(in_x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = (hx, cx)
        return m_output, rnn_hidden

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out size torch.Size([128, 16, 32, 32])
        for i in range(self.num_big_block):
            for j in range(self.num_blocks[i]):
                layer_i = self.layer_list[i]
                shortcut_i = self.shortcut_list[i]
                # print('layer i=0,j=0', layer_i[j])
                # print('big_block%d| layer%d| rnn%d: %s|' % (i, j, i, str(self.rnn_list[i])))
                # print('out size', out.size())
                if j == 0:
                    res = shortcut_i[j](out)
                else:
                    if j == 1:
                        rnn_hidden = self.set_m_rnn(out, self.rnn_memory_size_list[i])
                    bsz, channel, height, width = out.size()
                    m_out, rnn_hidden = self.m_rnn(out, self.rnn_list[i], rnn_hidden)
                    if self.m_out_list[i] is not None:
                        m_out = self.m_out_list[i](m_out)
                    m_out = torch.reshape(m_out, (bsz, height, width, channel)).permute((0, 3, 1, 2))
                    res = m_out
                out = layer_i[j](out)  # [bsz,dim,h,w]
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


class LmRnnKbSmallCIFAR10(nn.Module):
    # keep batch size same as origin, 32*32*16 ,16*16*32 8*8*64  as the input_size  can pass hidden or not pass hidden
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(LmRnnKbSmallCIFAR10, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.num_blocks = num_blocks
        self.num_big_block = len(num_blocks)

        self.layer_list = nn.ModuleList()
        self.shortcut_list = nn.ModuleList()
        self.rnn_list = nn.ModuleList()
        self.m_out_list = nn.ModuleList()
        self.rnn_memory_size_list = []
        self.convs_list = nn.ModuleList()
        self.deconvs_list = nn.ModuleList()

        self.args = args
        self.memory_type = args.memory_type
        self.pass_hidden = args.pass_hidden
        # self.keep_block_residual = args.keep_block_residual
        self.rnn_ratio = args.rnn_ratio
        self.num_downs = args.num_downs
        self.down_rate = 4 ** self.num_downs

        layer1, shortcut1, rnn1, m_out_linear1, rnn_memory_size1, convs1, deconvs1 = self._make_layer(block, 16, num_blocks[0], stride=1, fm=32)
        layer2, shortcut2, rnn2, m_out_linear2, rnn_memory_size2, convs2, deconvs2 = self._make_layer(block, 32, num_blocks[1], stride=2, fm=16)
        layer3, shortcut3, rnn3, m_out_linear3, rnn_memory_size3, convs3, deconvs3 = self._make_layer(block, 64, num_blocks[2], stride=2, fm=8)

        self.layer_list.extend([layer1, layer2, layer3])
        self.shortcut_list.extend([shortcut1, shortcut2, shortcut3])
        self.rnn_list.extend([rnn1, rnn2, rnn3])
        self.m_out_list.extend([m_out_linear1, m_out_linear2, m_out_linear3])
        self.rnn_memory_size_list.extend([rnn_memory_size1, rnn_memory_size2, rnn_memory_size3])
        self.convs_list.extend([convs1, convs2, convs3])
        self.deconvs_list.extend([deconvs1, deconvs2, deconvs3])

        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, fm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        shortcuts = nn.ModuleList()
        cur_fig_size = int(fm * fm / self.down_rate)
        rnn_input_size = block.expansion * planes * cur_fig_size
        rnn_memory_size = int(self.args.rnn_ratio * block.expansion * planes * cur_fig_size)
        if self.memory_type == 'rnn':
            rnn = torch.nn.RNNCell(rnn_input_size, rnn_memory_size, bias=True, nonlinearity='tanh')
        elif self.memory_type == 'lstm':
            rnn = torch.nn.LSTMCell(rnn_input_size, rnn_memory_size, bias=True)
        elif self.memory_type == 'gru':
            rnn = torch.nn.GRUCell(rnn_input_size, rnn_memory_size, bias=True)
        else:
            rnn = None
        if self.rnn_ratio != 1:
            m_out_linear = nn.Linear(rnn_memory_size, rnn_input_size)
        else:
            m_out_linear = None
        if self.num_downs > 0:
            convs = nn.ModuleList()
            deconvs = nn.ModuleList()
            for j in range(self.num_downs):
                convs.append(nn.Conv2d(in_channels=block.expansion*planes, out_channels=block.expansion*planes,
                                       kernel_size=3, stride=2, padding=1))
                deconvs.append(nn.ConvTranspose2d(block.expansion*planes, block.expansion*planes, kernel_size=3,
                                                  stride=2, padding=1))
        else:
            convs=None
            deconvs=None
        for i in range(num_blocks):
            # 对rnn来说，第一个残差连接虽然等维度，考虑到其他都是传h0，我就把当做和其他大块间残差的一样的
            stride = strides[i]
            # 16*16, 16*16 ..
            # 16*32(stride=2) 32*32 ..
            # 32*64(stride=2),64*64 ..
            layers.append(block(self.in_planes, planes, stride))
            if i == 0:
                if not self.pass_hidden:
                    shortcut = nn.Sequential()
                    if stride != 1 or self.in_planes != block.expansion * planes:
                        shortcut = nn.Sequential(
                            nn.Conv2d(self.in_planes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(block.expansion * planes)
                        )
                    shortcuts.append(shortcut)
                else:
                    # if self.keep_block_residual:
                    #     shortcut = nn.Sequential()
                    #     if stride != 1 or self.in_planes != block.expansion * planes:
                    #         shortcut = nn.Sequential(
                    #             nn.Conv2d(self.in_planes, block.expansion * planes, kernel_size=1, stride=stride,
                    #                       bias=False),
                    #             nn.BatchNorm2d(block.expansion * planes)
                    #         )
                    #     shortcuts.append(shortcut)
                    memory_shortcut = nn.Sequential()
                    if stride != 1 or self.in_planes != block.expansion * planes:
                        memory_shortcut = nn.Sequential(nn.Linear(rnn_memory_size*2, rnn_memory_size),
                                                 nn.BatchNorm2d(rnn_memory_size))
                    shortcuts.append(memory_shortcut)
            self.in_planes = planes * block.expansion
        return layers, shortcuts, rnn, m_out_linear, rnn_memory_size, convs, deconvs

    def set_m_rnn(self, x, rnn_memory_size):
        # origin_bsz, channel, height, width, = x.size()
        # bsz = height * width * origin_bsz
        bsz = x.size()[0]
        if self.memory_type in ['rnn',  'gru']:
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return hx
        if self.memory_type == 'lstm':
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                cx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return (hx, cx)

    def m_rnn(self, x, cur_i, rnn_hidden):
        input_size_list = []
        rnn = self.rnn_list[cur_i]
        if self.convs_list:
            # 可能四层dim变长的deconv更合理
            convs = self.convs_list[cur_i]
            for j in range(self.num_downs):
                input_size_list.append(x.size())
                x = convs[j](x)
        bsz, channel, new_height, new_width = x.size()
        x = x.permute([0, 2, 3, 1]).reshape(bsz, int(self.rnn_memory_size_list[cur_i]/ self.args.rnn_ratio))  # bsz, new_height * new_width * channel
        if self.memory_type in ['rnn', 'gru']:
            hx = rnn(x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = hx
        elif self.memory_type == 'lstm':
            hx, cx = rnn(x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = (hx, cx)
        if self.m_out_list[cur_i] is not None:
            m_output = self.m_out_list[cur_i](m_output)
        m_output = torch.reshape(m_output, (bsz,  new_height, new_height, channel,)).permute((0, 3, 1, 2))
        if self.deconvs_list:
            deconvs = self.deconvs_list[cur_i]
            for j in range(self.num_downs):
                m_output = deconvs[j](m_output, output_size=input_size_list[-j-1])
        return m_output, rnn_hidden

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out size torch.Size([128, 16, 32, 32])
        rnn_hidden = 0  # 0 to error
        for i in range(self.num_big_block):
            for j in range(self.num_blocks[i]):
                layer_i = self.layer_list[i]
                shortcut_i = self.shortcut_list[i]
                print('layer i=0,j=0', layer_i[j])
                print('big_block%d| layer%d| rnn%d: %s|' % (i, j, i, str(self.rnn_list[i])))
                print('out size', out.size())
                if not self.pass_hidden or i == 0:
                    if j == 0:
                        res = shortcut_i[j](out)
                    else:
                        if j == 1:
                            rnn_hidden = self.set_m_rnn(out, self.rnn_memory_size_list[i])
                        m_out, rnn_hidden = self.m_rnn(out, i, rnn_hidden)
                        res = m_out
                if self.pass_hidden and i > 0:
                    if j == 0:
                        print('shortcut_i[j]', shortcut_i[j])
                        rnn_hidden = shortcut_i[j](rnn_hidden)
                    m_out, rnn_hidden = self.m_rnn(out, i, rnn_hidden)
                    res = m_out
                out = layer_i[j](out)  # [bsz,dim,h,w]
                out += res
                out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def LmRnnKbSmall56CIFAR10(args):
    return LmRnnKbSmallCIFAR10(block=BaseBlock, num_blocks=[9, 9, 9], args=args)


def LmRnnKbSmall110CIFAR10(args):
    return LmRnnKbSmallCIFAR10(block=BaseBlock, num_blocks=[18, 18, 18], args=args)


class DepthTransposeCNN(nn.Module):
    def __init__(self,in_dim, out_dim, kernel_size=4, is_out=False):
        super(DepthTransposeCNN, self).__init__()
        self.nets = nn.ModuleList()
        self.is_out = is_out
        self.nets.extend([nn.ConvTranspose2d(in_channels=in_dim, out_channels=in_dim,
                                        kernel_size=kernel_size, stride=2, padding=1, groups=in_dim),
                          nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1,
                                        padding=0, bias=False)])
        if not is_out:
            self.nets.extend([nn.BatchNorm2d(in_dim),
                              nn.ReLU(True),
                              nn.BatchNorm2d(out_dim),
                              nn.ReLU(True)])

    def forward(self, x, output_size):
        bsz, dim, h, w = output_size
        if self.is_out:
            x = self.nets[0](x, output_size=[bsz, dim * 2, h, w])
            x = self.nets[1](x, output_size=output_size)
        else:
            x = self.nets[0](x, output_size=[bsz, dim * 2, h, w])
            x = self.nets[2](x)
            x = self.nets[3](x)
            x = self.nets[1](x, output_size=output_size)
            x = self.nets[4](x)
            x = self.nets[5](x)
        return x


class TransposeCNN(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, is_out=False):
        super(TransposeCNN, self).__init__()
        self.nets = nn.ModuleList()
        self.is_out = is_out
        self.nets.extend([nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim,
                                             kernel_size=kernel_size, stride=2, padding=1), ])
        if not is_out:
            self.nets.extend([nn.BatchNorm2d(out_dim),
                              nn.ReLU(True)])

    def forward(self, x, output_size):
        if self.is_out:
            x = self.nets[0](x, output_size=output_size)
        else:
            x = self.nets[0](x, output_size=output_size)
            x = self.nets[1](x)
            x = self.nets[2](x)
        return x


class LmRnnConsistentSmallCIFAR10(nn.Module):
    # keep batch size same as origin, 32*32*16 ,16*16*32 8*8*64  as the input_size  can pass hidden or not pass hidden
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(LmRnnConsistentSmallCIFAR10, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.num_blocks = num_blocks
        self.num_big_block = len(num_blocks)

        self.layer_list = nn.ModuleList()
        self.shortcut_list = nn.ModuleList()
        self.rnn_list = nn.ModuleList()
        self.m_out_list = nn.ModuleList()
        self.rnn_memory_size_list = []
        self.convs_list = nn.ModuleList()
        self.deconvs_list = nn.ModuleList()

        self.args = args
        self.memory_type = args.memory_type
        self.rnn_ratio = args.rnn_ratio
        self.conv_activate = args.conv_activate
        self.memory_before = args.memory_before
        self.depth_separate = args.depth_separate
        self.consistent_separate_rnn = args.consistent_separate_rnn
        self.dcgan_init = args.dcgan_init
        self.dcgan_kernel= args.dcgan_kernel
        self.dcgan_share_conv = args.dcgan_share_conv

        layer1, shortcut1, rnn1, m_out_linear1, rnn_memory_size1, convs1, deconvs1 = self._make_layer(block, 16, num_blocks[0], stride=1, fm=32,)
        layer2, shortcut2, rnn2, m_out_linear2, rnn_memory_size2, convs2, deconvs2 = self._make_layer(block, 32, num_blocks[1], stride=2, fm=16,)
        layer3, shortcut3, rnn3, m_out_linear3, rnn_memory_size3, convs3, deconvs3 = self._make_layer(block, 64, num_blocks[2], stride=2, fm=8,)

        self.layer_list.extend([layer1, layer2, layer3])
        self.shortcut_list.extend([shortcut1, shortcut2, shortcut3])
        if not self.consistent_separate_rnn:
            rnn2 = rnn1
            rnn3 = rnn1
        self.rnn_list.extend([rnn1, rnn2, rnn3])
        if not self.consistent_separate_rnn:
            m_out_linear2 = m_out_linear1
            m_out_linear3 = m_out_linear1
        self.m_out_list.extend([m_out_linear1, m_out_linear2, m_out_linear3])
        self.rnn_memory_size_list.extend([rnn_memory_size1, rnn_memory_size2, rnn_memory_size3])
        if self.dcgan_share_conv:
            # 32*32*16, 16*16*32, 8*8*64, 4*4*128,2*2*256,1*1*512
            # 1*1*512, 2*2*256, 4*4*128, 8*8*64, 16*16*32, 32*32*16,
            dim_list = [512, 256, 128, 64, 32, 16]
            convs2 = convs1[1:]
            convs3 = convs1[2:]
            deconvs2 = deconvs1[:-2].append(DepthTransposeCNN(in_dim=dim_list[-3], out_dim=dim_list[-2], kernel_size=self.dcgan_kernel,
                                         is_out=True) if self.depth_separate else TransposeCNN(in_dim=dim_list[-3], out_dim=dim_list[-2], kernel_size=self.dcgan_kernel, is_out=True))
            deconvs3 = deconvs2[:-3].append(DepthTransposeCNN(in_dim=dim_list[-4], out_dim=dim_list[-3], kernel_size=self.dcgan_kernel,
                                         is_out=True) if self.depth_separate else TransposeCNN(in_dim=dim_list[-4], out_dim=dim_list[-3], kernel_size=self.dcgan_kernel, is_out=True))

        self.convs_list.extend([convs1, convs2, convs3])
        self.deconvs_list.extend([deconvs1, deconvs2, deconvs3])
        if self.dcgan_init:
            self.deconvs_list.apply(self.weight_init)
            self.convs_list.apply(self.weight_init)

        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, fm, ):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        shortcuts = nn.ModuleList()
        down_rate = fm
        num_downs = int(np.log(fm)/np.log(2))
        cur_fig_size = int(fm * fm / down_rate)
        # build rnn
        rnn_input_size = block.expansion * planes * cur_fig_size
        rnn_memory_size = int(self.args.rnn_ratio * block.expansion * planes * cur_fig_size)
        assert rnn_memory_size == 512 * self.rnn_ratio
        if self.consistent_separate_rnn or fm ==32:
            if self.memory_type == 'rnn':
                rnn = torch.nn.RNNCell(rnn_input_size, rnn_memory_size, bias=True, nonlinearity='tanh')
            elif self.memory_type == 'lstm':
                rnn = torch.nn.LSTMCell(rnn_input_size, rnn_memory_size, bias=True)
            elif self.memory_type == 'gru':
                rnn = torch.nn.GRUCell(rnn_input_size, rnn_memory_size, bias=True)
            else:
                rnn = None
            # rnn out linear
            if self.rnn_ratio != 1:
                m_out_linear = nn.Linear(rnn_memory_size, rnn_input_size)
            else:
                m_out_linear = None
        else:
            rnn = None
            m_out_linear = None

        if self.conv_activate == 'lrelu':
            conv_activation = nn.LeakyReLU(True)
        elif self.conv_activate == 'relu':
            conv_activation = nn.ReLU(True)
        if num_downs > 0 or (self.dcgan_share_conv and fm != 32):
            dcgan_kernel=self.dcgan_kernel
            convs = nn.ModuleList()
            deconvs = nn.ModuleList()
            output_dim = block.expansion*planes
            for j in range(num_downs):
                output_dim = output_dim * 2
                # print('output_dim:', output_dim)
                if j == num_downs-1:
                    if self.depth_separate:
                        cur_conv = nn.Sequential(nn.Conv2d(in_channels=int(output_dim / 2), out_channels=int(output_dim / 2),
                                                           kernel_size=dcgan_kernel, stride=2, padding=1, groups=int(output_dim / 2)),
                                                 nn.Conv2d(in_channels=int(output_dim / 2), out_channels=output_dim, kernel_size=1, stride=1, padding=0, bias=False))
                    else:
                        cur_conv = nn.Sequential(nn.Conv2d(in_channels=int(output_dim/2), out_channels=output_dim,
                                                           kernel_size=dcgan_kernel, stride=2, padding=1))
                else:
                    if self.depth_separate:
                        cur_conv = nn.Sequential(nn.Conv2d(in_channels=int(output_dim / 2), out_channels=int(output_dim / 2),
                                                           kernel_size=dcgan_kernel, stride=2, padding=1, groups=int(output_dim / 2)),
                                                 nn.BatchNorm2d(int(output_dim / 2)),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(in_channels=int(output_dim / 2), out_channels=output_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                                 nn.BatchNorm2d(output_dim),
                                                 nn.ReLU(True))
                    else:
                        cur_conv = nn.Sequential(nn.Conv2d(in_channels=int(output_dim/2), out_channels=output_dim,
                                             kernel_size=dcgan_kernel, stride=2, padding=1),
                                             nn.BatchNorm2d(output_dim),
                                             conv_activation)
                convs.append(cur_conv)
            for j in range(num_downs):
                output_dim = int(output_dim / 2)
                # print('output_dim:',output_dim)
                if j == num_downs-1:
                    is_out = True
                else:
                    is_out = False
                if self.depth_separate:
                    cur_deconv = DepthTransposeCNN(in_dim=output_dim * 2, out_dim=output_dim, kernel_size=self.dcgan_kernel, is_out=is_out)
                else:
                    cur_deconv = TransposeCNN(in_dim=output_dim * 2, out_dim=output_dim, kernel_size=self.dcgan_kernel, is_out=is_out)
                deconvs.append(cur_deconv)
        else:
            convs=None
            deconvs=None
        for i in range(num_blocks):
            stride = strides[i]  # 16*16, 16*16 .. 16*32(stride=2) 32*32 .. 32*64(stride=2),64*64 ..
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
        return layers, shortcuts, rnn, m_out_linear, rnn_memory_size, convs, deconvs

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        return m

    def init_rnn_state(self, x, rnn_memory_size):
        # origin_bsz, channel, height, width, = x.size()
        # bsz = height * width * origin_bsz
        bsz = x.size()[0]
        if self.memory_type in ['rnn',  'gru']:
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return hx
        if self.memory_type == 'lstm':
                hx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                cx = torch.zeros(bsz, rnn_memory_size).cuda().type_as(x)
                return (hx, cx)

    def m_rnn(self, x, cur_i, rnn_hidden):
        input_size_list = []
        rnn = self.rnn_list[cur_i]
        num_downs = 5 - cur_i
        if self.convs_list:
            #  5,4,3,的deconv使得dim一致
            convs = self.convs_list[cur_i]
            for j in range(num_downs):
                input_size_list.append(x.size())
                # [128, 16, 32, 32]
                # [128,32,16,16]
                # [128,64, 8, 8]
                # [128,128,4,4]
                # [128, 256, 2, 2]
                # [128, 512, 1, 1]
                x = convs[j](x)
        bsz, channel, new_height, new_width = x.size()
        # print("self.convs_list[cur_i]",self.convs_list[cur_i])
        # print('after conv x size',x.size())
        x = x.permute([0, 2, 3, 1]).reshape(bsz, int(self.rnn_memory_size_list[cur_i] / self.args.rnn_ratio))  # bsz, new_height * new_width * channel
        if self.memory_type in ['rnn', 'gru']:
            hx = rnn(x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = hx
        elif self.memory_type == 'lstm':
            hx, cx = rnn(x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = (hx, cx)
        if self.m_out_list[cur_i] is not None:
            m_output = self.m_out_list[cur_i](m_output)
        m_output = torch.reshape(m_output, (bsz,  new_height, new_height, channel,)).permute((0, 3, 1, 2))
        if self.deconvs_list:
            deconvs = self.deconvs_list[cur_i]
            for j in range(num_downs):
                # print('j:%d deconv_in: %s| deconv j:%s' % (j, m_output.size(),deconvs[j]))
                m_output = deconvs[j](m_output, output_size=input_size_list[-j - 1])
        return m_output, rnn_hidden

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out size torch.Size([128, 16, 32, 32]) [128,32,16,16]
        rnn_hidden = 0  # 0 to error
        for i in range(self.num_big_block):
            for j in range(self.num_blocks[i]):
                layer_i = self.layer_list[i]
                # shortcut_i = self.shortcut_list[i]
                # print('layer i=0,j=0', layer_i[j])
                # print('big_block%d| layer%d| out size%s|' % (i, j, out.size()))
                if i == 0 and j == 0:
                    rnn_hidden = self.init_rnn_state(out, self.rnn_memory_size_list[i])
                if self.memory_before:
                    if j == 0:
                        m_in = self.shortcut_list[i][j](out)
                    else:
                        m_in =out
                    m_out, rnn_hidden = self.m_rnn(m_in, i, rnn_hidden)
                    res = m_out
                    out = layer_i[j](out)  # [bsz,dim,h,w]
                    out += res
                    out = F.relu(out)
                else:
                    out = layer_i[j](out)
                    m_out, rnn_hidden = self.m_rnn(out, i, rnn_hidden)
                    out += m_out
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def LmRnnConsistentSmall56CIFAR10(args):
    return LmRnnConsistentSmallCIFAR10(block=BaseBlock, num_blocks=[9, 9, 9], args=args)


def test():
    net = ResSmall20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
