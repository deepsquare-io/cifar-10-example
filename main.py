import argparse
import os

import horovod.torch as hvd
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms

import model

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--dataset", help="dataset root")
parser.add_argument("--checkpoint_in", help="checkpoint in")
parser.add_argument("--checkpoint_out", help="checkpoint out")
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--horovod",
    action="store_true",
    default=False,
    help="Enable distributed computing using Horovod",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="Disable CUDA for computing",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.horovod:
    hvd.init()

if args.cuda and args.horovod:
    torch.cuda.set_device(hvd.local_rank())

device = torch.device("cuda" if args.cuda else "cpu")
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
if args.horovod:
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, sampler=trainsampler
    )
else:
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
    )

testset = torchvision.datasets.CIFAR10(
    root=args.dataset, train=False, download=False, transform=transform_test
)
if args.horovod:
    testsampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, sampler=testsampler
    )
else:
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
    )

# Model
print("==> Building model...")

torch.set_float32_matmul_precision("high")
net = model.SimpleDLA()
BEST_ACC = 0.0
if not args.horovod:
    if args.cuda and torch.cuda.device_count() >= 1:
        print("Parallelized on", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    else:
        print("Parallelized on", torch.get_num_threads(), "threads!")
else:
    torch.set_num_threads(1)
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
if args.horovod:
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=net.named_parameters()
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if args.horovod:
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)


# Training
def train(epoch: int):
    print(f"\nEpoch: {epoch}")
    net.train()
    correct = 0
    if args.horovod:
        trainsampler.set_epoch(epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        if args.horovod:
            print(
                f"Rank: {hvd.rank()} | Train Epoch: {epoch} | Batch {batch_idx}/{len(trainloader)} | Loss: {loss.item():.6f} | Acc: {100.0 * correct / len(trainsampler):.3f}% ({correct}/{len(trainsampler)})",
            )
        else:
            print(
                f"Train Epoch: {epoch} | Batch {batch_idx}/{len(trainloader)} | Loss: {loss.item():.6f} | Acc: {100.0 * correct / len(batch_idx + 1):.3f}% ({correct}/{batch_idx + 1})",
            )


def test(epoch: int):
    global BEST_ACC
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            if args.horovod:
                print(
                    f"Rank: {hvd.rank()} | Epoch: {epoch} | Batch {batch_idx}/{len(testsampler)} | Loss: {test_loss / len(testsampler):.3f} | Acc: {100.0 * correct / len(testsampler):.3f}% ({correct}/{len(testsampler)})",
                )
            else:
                print(
                    f"Epoch: {epoch} | Batch {batch_idx}/{len(testloader)} | Loss: {test_loss / len(batch_idx + 1):.3f} | Acc: {100.0 * correct / len(batch_idx + 1):.3f}% ({correct}/{batch_idx + 1})",
                )

    # Save checkpoint.
    acc = 100.0 * correct / len(testsampler)
    if acc > BEST_ACC and args.checkpoint_out:
        if not args.horovod or (args.horovod and hvd.rank() == 0):
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, args.checkpoint_out)
            BEST_ACC = acc


for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
