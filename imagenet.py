
import torch.nn.parallel
import torch.optim
import torch.utils.data

from torch.autograd import Variable

from nasnet import NASNet, nasnetmobile, nasnetlarge, PowersignCD

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Trainer(object):
    cuda = torch.cuda.is_available()

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _loop(self, data_loader, is_train=True):
        loop_loss = []
        correct = []
        for data, target in tqdm(data_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=not is_train), Variable(target, volatile=not is_train)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data[0] / len(data_loader))
            correct.append((output.data.max(1)[1] == target.data).sum() / len(data_loader.dataset))
            if is_train:
                loss.backward()
                self.optimizer.step()
        mode = "train" if is_train else "test"
        print(f">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(correct):.2%}")
        return loop_loss, correct

    def train(self, data_loader):
        self.model.train()
        loss, correct = self._loop(data_loader)

    def test(self, data_loader):
        self.model.eval()
        loss, correct = self._loop(data_loader, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print(f"epochs: {ep}")
            self.train(train_data)
            self.test(test_data)
            if ep % self.save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir:
            name = f"weight-{epoch}-" + "-".join([f"{k}_{v}" for k, v in kwargs.items()]) + ".pkl"
            torch.save({"weight": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()},
                       os.path.join(self.save_dir, name))

def main(type, batch_size, data_root, n_epochs):
    if type == 'mobile':
        input_size = 224,
        model = nasnetmobile
    elif type == 'large':
        input_size = 331
        model = nasnetlarge
    else:
        input_size = 299
        model = nasnetlarge

    transform_train = transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    train = datasets.ImageFolder(traindir, transform_train)
    val = datasets.ImageFolder(valdir, transform_test)
    train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=True, num_workers=8)
    net = model(num_classes=1000)
    optimizer = PowersignCD(params=net.parameters(), steps=len(train)/batch_size*n_epochs, lr=0.6, momentum=0.9)
    trainer = Trainer(net, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(n_epochs, train_loader, test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--n_epochs", default=10, type=int)
    args = p.parse_args()
    main(args.batch_size, args.root, args.n_epochs)