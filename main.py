import argparse
import os
import torch.nn as nn
import torch.optim as optim

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import model

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--dataset", help="dataset root")
parser.add_argument("--checkpoint_in", help="checkpoint in")
parser.add_argument("--checkpoint_out", help="checkpoint out")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

trainset = torchvision.datasets.CIFAR10(
    root=args.dataset, train=True, download=False, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root=args.dataset, train=False, download=False, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# Model
print("==> Building model...")

torch.set_float32_matmul_precision("high")
net = model.SimpleDLA()
BEST_ACC = 0.0
if torch.cuda.device_count() >= 1:
    print("Parallelized on", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print("Parallelized on", torch.get_num_threads(), "threads!")
net = net.to(device)
net = torch.compile(net)
print("Model is JIT-compiling enabled!")

if args.checkpoint_in:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    if os.path.isfile(args.checkpoint_in):
        checkpoint = torch.load(args.checkpoint_in)
        net.load_state_dict(checkpoint["net"])
        BEST_ACC = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]
    else:
        print("no checkpoint found")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch: int):
    print(f"\nEpoch: {epoch}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(
            f"Epoch: {epoch} | Batch {batch_idx}/{len(trainloader)} | Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100.0 * correct / total:.3f}% ({correct}/{total})",
        )


def test(epoch: int):
    global BEST_ACC
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(
                f"Epoch: {epoch} | Batch {batch_idx}/{len(testloader)} | Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100.0 * correct / total:.3f}% ({correct}/{total})",
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > BEST_ACC and args.checkpoint_out:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        torch.save(state, args.checkpoint_out)
        BEST_ACC = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()
