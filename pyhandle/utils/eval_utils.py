import numpy as np
import torch
import torch.nn.functional as F

from utils.matplot_utils import plot_classes_preds


def eval_for_one_epoch(dataset, model, total_step=10):
    eval_result = {}

    model.eval()
    with torch.no_grad():
        eval_result['pred_figs'] = pred_and_draw_image(model, dataset)
        eval_result['precision'], eval_result['recall'] = metric_eval(total_step, model, dataset)

    return eval_result


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    cpu_preds = preds_tensor.cpu()
    preds = np.squeeze(cpu_preds.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def pred_and_draw_image(model, dataset):
    loader = dataset.test_loader
    images, labels = next(iter(loader))
    preds, probs = images_to_probs(model, images.cuda())
    return plot_classes_preds(preds, probs, images.cuda(), labels.cuda())


def metric_eval(total_step, model, dataset):
    TP = FP = TN = FN = 0
    loader = dataset.test_loader
    for images, labels in loader:
        preds = model(images.cuda())
        _, class_preds_batch = torch.max(preds, 1)

        np_labels = labels.numpy()
        np_preds = class_preds_batch.cpu().numpy()

        TP += sum(np_labels[np.where(np_preds == 1)[0]] == 1)
        FP += sum(np_labels[np.where(np_preds == 1)[0]] == 0)
        TN += sum(np_labels[np.where(np_preds == 0)[0]] == 0)
        FN += sum(np_labels[np.where(np_preds == 0)[0]] == 1)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        return precision, recall
