import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from mobilenet import MobileNet


#
parser = argparse.ArgumentParser(description="Pytorch CIFAR10 Training")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--model', default="mobilenet", type=str,
                   help='model type (defualt: Mobilenet)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (defualt: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (defualt: 1)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0
start_epoch = 0

if args.seed != 0:
    torch.manual_seed(args.seed)


# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform_train)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root='./data', train=False, download=False,
                            transform=transform_test)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=8)

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name+'_'+str(args.seed))

    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

else:
    print('==> Building model..')
    net = MobileNet()

if not os.path.isdir('results'):
    os.mkdir('results')

logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    netr = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print("Using CUDA..")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5, verbose=True
)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # batch_size만큼의 random index 1차원 벡터\

    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEPoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).cpu().sum().float()
                    + (1-lam) * predicted.eq(targets_b).cpu().sum().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f" train_loss:{train_loss / batch_idx}  |  accuracy:{ 100. * correct / total}")

    return (train_loss / batch_idx, reg_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            print(f" test_loss:{test_loss / batch_idx}  |  accuracy:{100. * correct / total}")

        acc = 100. * correct / total
        if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc
        return (test_loss / batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    print('Saving..')
    state = {
        'net':  net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step(train_loss)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])

