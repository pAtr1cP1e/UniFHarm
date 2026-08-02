import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler
from matplotlib import ticker
from scipy.odr import Model
from transformers import get_cosine_schedule_with_warmup
from DataProcess import *
from Model import *
from evaluate import *
import time
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm as cm
import matplotlib as mpl


def train(task='dti', dataset_name='biosnap', cold=None, batch_size=128, lr=1e-4, epochs=30):
    print(f'start training <{task}> task on <{dataset_name}>, cold->{cold}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(5):
        train_loader, valid_loader, test_loader = get_dataloader(
            task=task,
            batch_size=batch_size,
            name=dataset_name,
            fold=i,
            cold=cold
        )

        model = DownStreamModel(task=task, device=device).to(device)

        trainable_params = [
            param for param in model.parameters()
            if param.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=3e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6, last_epoch=-1
        )

        best_valid_loss = 1000
        best_epoch = 0
        log_time = datetime.datetime.now().strftime('%m%d-%H%M')
        model.train()

        for epoch in range(epochs):
            time_start = time.time()
            loss_epoch = 0
            for index, batch in enumerate(train_loader):
                optimizer.zero_grad()
                train_loss, _, _, _ = model(batch)

                loss_epoch += train_loss.item()
                train_loss.backward()
                optimizer.step()

                if index % 50 == 0:
                    time_used = (((time.time() - time_start) / (index + 1)) * (len(train_loader) - index)) / 60
                    print(f'Epoch {epoch}, iter {index}/{len(train_loader)} ,'
                          f'loss: {train_loss.item():.4f}, avg loss: {loss_epoch / (index + 1):.3f}, '
                          f'lr:{scheduler.get_last_lr()[0]:.1e}, '
                          f'remaining time/epoch: {time_used:.1f} min')
                    with open(f'./logs/{task.upper()}/{dataset_name}_fold{i}_{log_time}.txt', 'a') as f:
                        f.write(f'Epoch {epoch}, iter {index}/{len(train_loader)}, '
                                f'loss: {train_loss.item():.3f}, avg loss: {loss_epoch / (index + 1):.3f}\n'
                                )

            print('=====Validating=====')
            valid_loss, res_metrics = validation(model, valid_loader, task=task)
            print(f'valid loss:{valid_loss:.3f}, best loss:{best_valid_loss:.3f}, {res_metrics}')
            with open(f'./logs/{task.upper()}/{dataset_name}_fold{i}_{log_time}.txt', 'a') as f:
                f.write(f'valid loss:{valid_loss:.3f}, best loss:{best_valid_loss:.3f}, {str(res_metrics)}\n')

            print('=====Evaluating=====')
            test_loss, res_metrics = validation(model, test_loader, task=task)
            print(f'test loss:{test_loss:.3f}, {res_metrics}')
            with open(f'./logs/{task.upper()}/{dataset_name}_fold{i}_{log_time}.txt', 'a') as f:
                f.write(f'test loss:{test_loss:.3f}, {str(res_metrics)}\n')

            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch

            # if epoch - best_epoch >= 10:
            #     print(f'not improvement from lowest valid loss epoch {best_epoch}')
            #     break

            model.train()
            scheduler.step()

        print(f'best valid loss -> {best_valid_loss}, epoch -> {best_epoch}')
        with open(f'./logs/{task.upper()}/{dataset_name}_fold{i}_{log_time}.txt', 'a') as f:
            f.write(f'best valid loss -> {best_valid_loss}, epoch -> {best_epoch}')
        break


def validation(model, valid_loader, task='dti'):
    res, valid_loss = None, -1.0
    model.eval()
    if task in ['dti', 'moa']:
        probs, preds, labels, valid_loss = dti_test(model, valid_loader)
        roc_auc, pr_auc, ap, precision, recall, accuracy, f1 = dti_metrics(probs, preds, labels)
        res = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ap': ap, 'precision': precision,
               'recall': recall, 'accuracy': accuracy, 'f1': f1}
        with open('biosnap_preds.txt', 'a') as f:
            for pred in preds:
                f.write(f'{int(pred)}\n')
    else:
        preds, labels, valid_loss = dta_test(model, valid_loader)
        mse, rmse, rm2, pearson, spearman, cindex, aupr = dta_metrics(preds, labels)
        res = {'mse': mse, 'rmse': rmse, 'rm2': rm2, 'pearson': pearson,
               'spearman': spearman, 'cindex': cindex, 'aupr': aupr}

    return valid_loss, res


if __name__ == '__main__':
    train(task='dti', dataset_name='biosnap', cold=None, batch_size=16, lr=1e-4, epochs=20)
    # train(task='dta', dataset_name='davis', cold=None, batch_size=16, lr=1e-4, epochs=20)
    pass


