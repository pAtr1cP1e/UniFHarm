import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DTIDataset(Dataset):
    def __init__(self, path='./data/DTI', dataset_name='human',
                 fold=0, state='train', cold=None):
        self.data = self.load_data(
            path=path, dataset_name=dataset_name,
            fold=fold, state=state, cold=cold
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'mol': self.data[idx]['smiles'],
            'prot': self.data[idx]['protein'],
            'label': torch.tensor(self.data[idx]['label'], dtype=torch.long),
        }

    def load_data(self, path='./data/DTI', dataset_name='human',
                  fold=0, state='train', cold=None):
        if not cold:
            dataset_path = os.path.join(path, dataset_name, 'folds', f'{state}_fold_{fold}.csv')
        else:
            dataset_path = os.path.join(path, dataset_name, 'cold', f'un_{cold}', f'{state}_fold_{fold}.csv')
        df = pd.read_csv(dataset_path)
        return df.to_dict(orient='records')


class DTADataset(Dataset):
    def __init__(self, path='./data/DTA', dataset_name='davis',
                 fold=0, state='train', cold=None):
        self.data = self.load_data(
            path=path, dataset_name=dataset_name,
            fold=fold, state=state, cold=cold
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'mol': self.data[idx]['mol_seq'],
            'prot': self.data[idx]['protein_seq'],
            'label': torch.tensor(self.data[idx]['affinity'], dtype=torch.float32)
        }

    def load_from_csv(self, path):
        df = pd.read_csv(path)
        return df.to_dict(orient='records')

    def load_data(self, path='./data/DTA', dataset_name='davis', fold=0, state='train', cold=None):
        if not cold:
            path = os.path.join(path, dataset_name, 'folds', f'{state}_fold_{fold}.csv')
            data = self.load_from_csv(path)
        else:
            path = os.path.join(path, dataset_name, 'cold', f'un_{cold}', f'{state}_fold_{fold}.csv')
            data = self.load_from_csv(path)
        return data


def get_dataloader(task='dti', batch_size=32, name='human', fold=0, cold=None):
    if task == 'dti':
        train_set = DTIDataset(dataset_name=name, fold=fold, state='train', cold=cold)
        valid_set = DTIDataset(dataset_name=name, fold=fold, state='valid', cold=cold)
        test_set = DTIDataset(dataset_name=name, fold=fold, state='test', cold=cold)
    elif task == 'dta':
        train_set = DTADataset(dataset_name=name, fold=fold, state='train', cold=cold)
        valid_set = DTADataset(dataset_name=name, fold=fold, state='valid', cold=cold)
        test_set = DTADataset(dataset_name=name, fold=fold, state='test', cold=cold)
    else:
        pass
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader
