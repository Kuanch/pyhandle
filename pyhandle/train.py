import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from net import network
from dataset import dataloader
from utils.utils import plot_classes_preds, images_to_probs


def stdout_statistic(epoch, step, running_loss):
    # print statistics
    if step % 10 == 9:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 10))


def eval_for_one_epoch(dataset, model, total_step=10):
    eval_result = {}

    with torch.no_grad():
        eval_result['pred_figs'] = pred_and_draw_image(model, dataset)
        eval_result['precision'], eval_result['recall'] = metric_eval(total_step, model, dataset)

    return eval_result


def pred_and_draw_image(model, dataset):
    images, labels = dataset.read_test()
    preds, probs = images_to_probs(model, images)
    return plot_classes_preds(preds, probs, images, labels)


def metric_eval(total_step, model, dataset):
    TP = FP = TN = FN = 0
    for step in range(total_step):
        images, labels = dataset.read_test()
        preds = model(images)
        _, class_preds_batch = torch.max(preds, 1)

        np_labels = labels.cpu().numpy()
        np_preds = class_preds_batch.cpu().numpy()

        TP += sum(np_labels[np.where(np_preds == 1)[0]] == 1)
        FP += sum(np_labels[np.where(np_preds == 1)[0]] == 0)
        TN += sum(np_labels[np.where(np_preds == 0)[0]] == 0)
        FN += sum(np_labels[np.where(np_preds == 0)[0]] == 1)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        return precision, recall


def train_loop(model, criterion, optimizer, dataset, writer, epoch):
    total_step = len(dataset.train_loader)
    for e in range(epoch):
        running_loss = 0.
        for step in range(total_step):
            inputs, labels = dataset.read_train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, step)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            stdout_statistic(e, step, running_loss)
            running_loss = 0.
            del loss

        eval_result = eval_for_one_epoch(dataset, model)
        writer.add_figure('predictions vs. actuals', eval_result['pred_figs'], global_step=e)
        writer.add_scalar("Val/precision", eval_result['precision'], e)
        writer.add_scalar("Val/recall", eval_result['recall'], e)
    writer.flush()
    writer.close()


def training_setup(args):
    training_setup = {}
    training_setup['model'] = network.get_classifier(args.model_name, num_classes=10).cuda()
    training_setup['dataset'] = dataloader.TorchLoader(args.dataset_name, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    training_setup['criterion'] = nn.CrossEntropyLoss()
    training_setup['optimizer'] = optim.SGD(training_setup['model'].parameters(), lr=0.001, momentum=0.9)
    training_setup['writer'] = SummaryWriter()

    return training_setup


def train(args):
    setup = training_setup(args)
    train_loop(**setup, epoch=args.training_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--dataset_name', default='CIFAR10')
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--training_epoch', type=int, default=1)
    parser.add_argument('--eval_step', type=int, default=10000)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    train(args)
