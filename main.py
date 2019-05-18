'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import codecs
import time
from utils import format_time
from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--model', '-model', default='ResNet34', type=str, help='ResNet34, RfNet34, Lm_Rnn_Net34')
parser.add_argument('--gpu', '-gpu', type=int,default=0, help='')
parser.add_argument('--epoch', '-epoch', type=int, default=350, help='')
parser.add_argument('--log', '-log',  type=str, default='resnet34', help='')

parser.add_argument('--memory_type', type=str, default='no', help='no,rnn,gru,lstm,san,rmc,dnc')
parser.add_argument('--dim_type', type=str, default='hw,channel', help='')
parser.add_argument('--include_last', type=int, default=1, help='')

# RMC
parser.add_argument('--memslots', type=int, default=4, help='')
parser.add_argument('--numheads', type=int, default=4, help='')
parser.add_argument('--headsize', type=int, default=128, help='')
parser.add_argument('--numblocks', type=int, default=1, help='')
parser.add_argument('--forgetbias', type=int, default=1, help='')
parser.add_argument('--inputbias', type=int, default=0, help='')
parser.add_argument('--attmlplayers', type=int, default=2, help='')
parser.add_argument('--outmode', type=str, default='nextmemory', help='nextmemory,ouput_x')

#DNC
parser.add_argument('-input_size', type=int, default=6, help='dimension of input feature')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=64, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')
parser.add_argument('-nlayer', type=int, default=1, help='number of layers')
parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
# parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')
parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=20, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=16, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')
parser.add_argument('-sequence_max_length', type=int, default=1000, metavar='N', help='sequence_max_length')
# parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('-iterations', type=int, default=2000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')
parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')
#
#
#SAN
parser.add_argument('--head_count', type=int, default=4, help='')
parser.add_argument('--exclude_self',type=bool,default=False)
parser.add_argument('--ex_vw',default=False,type=bool)
#
# # RNN
parser.add_argument('--pass_hidden', type=int, default=0, help='pass hidden states through shortcuts')
parser.add_argument('--rnn_res', type=int, default=0, help='rnn like res')
parser.add_argument('--memory_position', type=str, default='before', help='before,after')
parser.add_argument('--rnn_ratio', type=float, default=1, help='rnn_memory_size= rnn_ratio*emb_dim')
parser.add_argument('--rnn_init_type', type=str, default='zeros')
parser.add_argument('--rnn_integrate_type', type=str, default='add', help='add,concat_linear,update')
args = parser.parse_args()
# if not os.path.isdir('logs'):
#     os.mkdir('logs')
log_dir = 'checkpoint/' + args.log + '.txt'
f_log = codecs.open(filename=log_dir, mode='w+')
print(args)
print(args, file=f_log)
torch.cuda.set_device(args.gpu)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
print('==> Preparing data..', file=f_log)
# f_log.write('==> Preparing data..\n')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
print('==> Building model..', file=f_log)
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

if args.model == 'ResNet34':
    net = ResNet34()
elif args.model == 'ResNet101':
    net = ResNet101()
elif args.model == 'RfNet34':
    net = RfNet34(args)
elif args.model == 'LmRnnNet34':
    net = LmRnnNet34(args)
elif args.model == 'PreActResNet18':
    net = PreActResNet18()

elif args.model == 'ResSmall56':
    net=ResSmall56()
elif args.model == 'ResSmall110':
    net=ResSmall110()
elif args.model == 'RfSmall56':
    net = RfSmall56(args)
elif args.model == 'RfSmall110':
    net = RfSmall110(args)
elif args.model == 'LmRnnSmall56':
    net=LmRnnSmall56(args)
elif args.model == 'LmRnnSmall110':
    net = LmRnnSmall110(args)
net = net.cuda()
# net = net.to(args.gpu)
print(net)
print(net, file=f_log)
print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in net.parameters()),
            sum(p.numel() for p in net.parameters() if p.requires_grad),
        ))
print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in net.parameters()),
            sum(p.numel() for p in net.parameters() if p.requires_grad),
        ), file=f_log)
net = torch.nn.DataParallel(module=net, device_ids=[args.gpu])
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 250], gamma=0.1)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = inputs.to(args.gpu), targets.to(args.gpu)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_acc = 100.*correct/total
    return train_acc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets=inputs.to(args.gpu), targets.to(args.gpu)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving, best test acc:%.3f' % acc)
        print('Saving, best test acc:%.3f' % acc, file=f_log)
        # f_log.write('Saving, best test acc' + str(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('./checkpoint/'+args.log):
            os.mkdir('./checkpoint/'+args.log)
        torch.save(state, './checkpoint/'+ args.log + os.sep + 'epoch:'+str(epoch)+'acc:%.3f' % acc)
        best_acc = acc
    return acc


# f_log.write(str(args))
for epoch in range(start_epoch, start_epoch+args.epoch):
    train_begin_time = time.time()
    train_acc = train(epoch)
    train_end_time = time.time()
    test_acc = test(epoch)
    # test_end_time = time.time()

    tot_time = train_end_time - train_begin_time
    print('cur_lr'+str(scheduler.get_lr()) + 'epoch:' + str(epoch) + '| train acc:%.3f' % train_acc + '| test_acc:%.3f' % test_acc + '| Train time: %s' % format_time(tot_time))
    print('cur_lr'+str(scheduler.get_lr()) + 'epoch:' + str(epoch) + '| train acc:%.3f' % train_acc + '| test_acc:%.3f' % test_acc + '| Train time: %s' % format_time(tot_time), file=f_log)
    scheduler.step(epoch)
f_log.close()
