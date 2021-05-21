import torch
import torch.optim as optim
from wideresnet import WideResNet
from data import get_dataloaders
from smooth_ce import SmoothCrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.1
batch_size = 32
weight_decay = 5e-4
num_epochs = 200
num_classes = 10
train_step = 0
val_step = 0
test_step = 0

def train(trainloader, model, optimizer, criterion, writer, scheduler):
    global train_step
    model.train()
    losses = []
    accuracies = []
    loop = tqdm(trainloader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        num_correct = (predicted == y).sum()
        accuracy = float(num_correct) / float(x.shape[0]) * 100
        accuracies.append(accuracy)

        writer.add_scalar('Training loss', loss, global_step=train_step)
        writer.add_scalar('Training Accuracy', accuracy, global_step=train_step)
        loop.set_postfix(Train_loss=loss.item(), Train_accuracy=f"{sum(accuracies)/len(accuracies):.1f}")
        train_step += 1
    scheduler.step(sum(losses)/len(losses))

def validation(validloader, model, criterion, writer):
    global val_step
    model.eval()
    losses = []
    accuracies = []
    loop = tqdm(validloader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss.item())

        predicted = torch.argmax(outputs, dim=1)
        num_correct = (predicted == y).sum()
        accuracy = float(num_correct) / float(x.shape[0]) * 100
        accuracies.append(accuracy)
        writer.add_scalar('Validation loss', loss, global_step=val_step)
        writer.add_scalar('Validation Accuracy', accuracy, global_step=val_step)

        loop.set_postfix(Val_loss=loss.item(), Val_accuracy=f"{sum(accuracies)/len(accuracies):.1f}")
        val_step += 1


def test(testloader, model, criterion, writer):
    global test_step
    model.eval()
    losses = []
    accuracies = []
    loop = tqdm(testloader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss.item())

        predicted = torch.argmax(outputs, dim=1)
        num_correct = (predicted == y).sum()
        accuracy = float(num_correct) / float(x.shape[0]) * 100
        accuracies.append(accuracy)

        writer.add_scalar('Test loss', loss, global_step=test_step)
        writer.add_scalar('Test Accuracy', accuracy, global_step=test_step)
        loop.set_postfix(Test_loss=loss.item(), Test_accuracy=f"{sum(accuracies)/len(accuracies):.1f}")
        test_step += 1


def main():

    trainsampler, trainloader, validloader, testloader = get_dataloaders(batch_size, 'dataset', 0.15)

    model = WideResNet(10, 2, 0, 10)  # depth, widen_factor, dropout_rate, num_classes
    model.to(device)

    criterion = SmoothCrossEntropyLoss(0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
    writer = SummaryWriter(f'runs/Rand_Augmentation')

    for epoch in range(num_epochs):
        print(f"Epoch:{epoch+1}")
        train(trainloader, model, optimizer, criterion, writer, scheduler)
        validation(validloader, model, criterion, writer)

        if (epoch+1) % 10 == 0:
            test(testloader, model, criterion, writer)


if __name__ == '__main__':
    main()


