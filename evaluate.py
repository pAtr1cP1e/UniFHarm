import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import average_precision_score, precision_score
from sklearn.metrics import recall_score, accuracy_score, f1_score
import torch
from math import sqrt
from scipy import stats


def get_roc_auc(probs, labels):
    return roc_auc_score(labels, probs)


def get_pr_auc(probs, labels):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_average_precision(probs, labels):
    return average_precision_score(labels, probs)


def get_precision(preds, labels):
    return precision_score(labels, preds, average='binary')


def get_recall(preds, labels):
    return recall_score(labels, preds, average='binary')


def get_accuracy(preds, labels):
    return accuracy_score(labels, preds)


def get_f1(preds, labels):
    return f1_score(labels, preds, average='binary')


def dti_test(model, loader):
    probs_list = []
    preds_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in loader:
            valid_loss, _, probs, preds = model(batch)
            total_loss += valid_loss.item()
            probs_list.extend(probs.detach().cpu().numpy())
            preds_list.extend(preds.detach().cpu().numpy())
            labels_list.extend(batch['label'].detach().cpu().numpy())
    return np.array(probs_list), np.array(preds_list), np.array(labels_list), total_loss / len(loader)


def dti_metrics(probs, preds, labels):
    roc_auc = get_roc_auc(probs, labels)  # AUC
    pr_auc = get_pr_auc(probs, labels)  # PRAUC
    ap = get_average_precision(probs, labels)  # AUPRC
    precision = get_precision(preds, labels)  # precision
    recall = get_recall(preds, labels)
    accuracy = get_accuracy(preds, labels)
    f1 = get_f1(preds, labels)
    return roc_auc, pr_auc, ap, precision, recall, accuracy, f1


# evaluate dta
def get_mse(preds, labels):
    return np.mean((labels - preds) ** 2)


def get_pearson(preds, labels):
    return np.corrcoef(labels, preds)[0, 1]


def get_spearman(preds, labels):
    return stats.spearmanr(labels, preds)[0]


def get_cindex(preds, labels):
    ind = np.argsort(labels)
    labels = labels[ind]
    preds = preds[ind]
    i = len(labels) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if labels[i] > labels[j]:
                z = z + 1
                u = preds[i] - preds[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def get_rm2(preds, labels):
    def r_squared_error(y_obs, y_pred):
        y_obs = np.array(y_obs)
        y_pred = np.array(y_pred)
        y_obs_mean = [np.mean(y_obs) for y in y_obs]
        y_pred_mean = [np.mean(y_pred) for y in y_pred]

        mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
        mult = mult * mult

        y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
        y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

        return mult / float(y_obs_sq * y_pred_sq)

    def squared_error_zero(y_obs, y_pred):
        def get_k(y_obs, y_pred):
            y_obs = np.array(y_obs)
            y_pred = np.array(y_pred)

            return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

        k = get_k(y_obs, y_pred)

        y_obs = np.array(y_obs)
        y_pred = np.array(y_pred)
        y_obs_mean = [np.mean(y_obs) for y in y_obs]
        upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
        down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

        return 1 - (upp / float(down))

    r2 = r_squared_error(labels, preds)
    r02 = squared_error_zero(labels, preds)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_aupr(preds, labels, threshold=7.0):
    labels = np.where(labels >= threshold, 1, 0)
    preds = np.where(preds >= threshold, 1, 0)
    aupr = average_precision_score(labels, preds)
    return aupr


def dta_test(model, loader):
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in loader:
            tl, pred, _, _ = model(batch)
            total_loss += tl.item()
            predictions.extend(pred.detach().cpu().numpy().flatten().tolist())
            labels.extend(batch['label'].detach().cpu().numpy().flatten().tolist())
    return np.array(predictions), np.array(labels), total_loss / len(loader)


def dta_metrics(preds, labels):
    mse = get_mse(preds, labels)
    rmse = sqrt(mse)
    rm2 = get_rm2(preds, labels)
    pearson = get_pearson(preds, labels)
    spearman = get_spearman(preds, labels)
    cindex = get_cindex(preds, labels)
    aupr = get_aupr(preds, labels)
    return mse, rmse, rm2, pearson, spearman, cindex, aupr

