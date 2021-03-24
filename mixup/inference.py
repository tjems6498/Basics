import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils import data_loader
from mobilenet import MobileNet


import pdb


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
trainloader, testloader = data_loader(8)
net = MobileNet().cuda()

if os.path.isdir("checkpoint"):
    path = "checkpoint/ckpt.t70_20210324"
    checkpoint = torch.load(path)

    net.load_state_dict(checkpoint['net'])

#
# with torch.no_grad():
#     total = 0
#     correct = 0
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         outputs = net(inputs)
#
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()
#
#         print((100. * correct / total).item())
#         break


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=12):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.figure(figsize=(20, 20))
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(classes[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



if __name__ == '__main__':
    visualize_model(net, num_images=12)