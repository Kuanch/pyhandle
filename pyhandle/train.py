import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from net import network
from dataset import dataloader
from utils.eval_utils import eval_for_one_epoch


def train_for_one_step(model, criterion, optimizer, dataset, loss_container):
    inputs, labels = dataset.read_train()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loss_container[0] = loss.item()

    del loss


def train_for_one_epoch(model, criterion, optimizer, dataset, epoch, writer=None):
    losses = [0.]
    step_per_epoch = len(dataset.train_loader)
    step_to_draw_loss = int(step_per_epoch / 1000) + 1
    for step in range(step_per_epoch):
        train_for_one_step(model, criterion, optimizer, dataset, losses)

        if writer is not None and step % step_to_draw_loss == 0:
            writer.add_scalar("Loss/train", losses[0], step + epoch * step_per_epoch)


def train_loop(training_setup, epoch):
    writer = training_setup['writer']
    for e in range(epoch):
        train_for_one_epoch(**training_setup, epoch=e)

        eval_result = eval_for_one_epoch(training_setup['dataset'], training_setup['model'])
        writer.add_figure('predictions vs. actuals', eval_result['pred_figs'], global_step=e)
        writer.add_scalar("Val/precision", eval_result['precision'], e)
        writer.add_scalar("Val/recall", eval_result['recall'], e)
    writer.flush()
    writer.close()


def training_setup(args):
    training_setup = {}
    training_setup['model'] = network.get_classifier(args.model_name, num_classes=args.num_classes).cuda()
    training_setup['dataset'] = dataloader.TorchLoader(args.dataset_name, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    training_setup['criterion'] = nn.CrossEntropyLoss()
    training_setup['optimizer'] = optim.SGD(training_setup['model'].parameters(), lr=0.001, momentum=0.9)
    training_setup['writer'] = SummaryWriter()

    return training_setup


def train(args):
    setup = training_setup(args)
    train_loop(setup, epoch=args.training_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--dataset_name', default='CIFAR10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--training_epoch', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    train(args)
