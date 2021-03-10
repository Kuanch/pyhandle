import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from net import network
from dataset import dataloader
from optim.get_optimizer import get_optim
from utils.eval_utils import eval_for_one_epoch
from utils.pytorch_utils import save_model


def train_for_one_step(model, criterion, optimizer, inputs, labels, loss_container):
    # zero the parameter gradients
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None

    # forward + backward + optimize
    outputs = model(inputs.cuda())

    loss = criterion(outputs, labels.cuda())
    loss.backward()
    optimizer.step()
    loss_container[0] = loss.item()

    del loss


def train_for_one_epoch(model, criterion, optimizer, dataset, epoch, writer=None):
    losses = [0.]
    step = 0
    step_per_epoch = len(dataset.train_loader)
    step_to_draw_loss = int(step_per_epoch / 100) + 1
    for images, labels in dataset.train_loader:
        train_for_one_step(model, criterion, optimizer, images, labels, losses)

        if writer is not None and step % step_to_draw_loss == 0:
            writer.add_scalar("Loss/train", losses[0], step + epoch * step_per_epoch)

        step += 1


def train_loop(training_setup, epoch):
    model = training_setup['model']
    criterion = training_setup['criterion']
    optim = training_setup['optimizer']
    dataset = training_setup['dataset']
    writer = training_setup['writer']

    for e in range(epoch):
        model.train()
        train_for_one_epoch(model, criterion, optim, dataset, epoch=e, writer=writer)

        eval_result = eval_for_one_epoch(training_setup['dataset'], training_setup['model'])
        writer.add_figure('predictions vs. actuals', eval_result['pred_figs'], global_step=e)
        writer.add_scalar("Val/precision", eval_result['precision'], e)
        writer.add_scalar("Val/recall", eval_result['recall'], e)
    writer.flush()
    writer.close()

    if training_setup['save_model_path'] is not None:
        path = training_setup['save_model_path']
        model_state_dict = training_setup['model'].state_dict()
        optimizer_state_dict = training_setup['optimizer'].state_dict()
        loss = training_setup['criterion']
        save_model(model_state_dict, optimizer_state_dict, loss, epoch, path)


def training_setup(args):
    training_setup = {}
    training_setup['model'] = network.get_classifier(args.model_name, num_classes=args.num_classes).cuda()
    training_setup['dataset'] = dataloader.TorchLoader(args.dataset_name, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    training_setup['criterion'] = nn.CrossEntropyLoss()
    training_setup['optimizer'] = get_optim('SGD', training_setup['model'].parameters(), args.lr)
    training_setup['writer'] = SummaryWriter()
    training_setup['save_model_path'] = args.save_model_path

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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--save_model_path', type=str, default=None)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    train(args)
