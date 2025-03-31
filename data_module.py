import os
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from datasets import QM9

# Assume QM9 is defined as in the previous example.
# from your_dataset_file import QM9

class QM9DataModule(pl.LightningDataModule):
    def __init__(self,transform, root: str = "./data/QM9", batch_size: int = 32, 
                 train_val_test_split: tuple = (0.8, 0.1, 0.1), num_workers: int = 4):
        """
        DataModule for the QM9 dataset.
        
        Args:
            root (str): Directory where the dataset is stored/downloaded.
            batch_size (int): Batch size for dataloaders.
            train_val_test_split (tuple): Proportions for train, val, and test splits.
            num_workers (int): Number of workers for dataloaders.
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.transform = transform
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # This method is called only on one GPU/CPU and is meant to download data.
        # Instantiate the dataset; this will trigger the download and processing if needed.
        QM9(self.root,transform=self.transform)

    def setup(self, stage=None):
        # Called on every GPU: split data, apply transforms, etc.
        if self.dataset is None:
            self.dataset = QM9(self.root)
        
        # Get total number of samples.
        total_samples = len(self.dataset)
        
        # Calculate lengths for train/val/test splits.
        train_len = int(total_samples * self.train_val_test_split[0])
        val_len = int(total_samples * self.train_val_test_split[1])
        test_len = total_samples - train_len - val_len
        
        # Use random_split to divide the dataset.
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, lengths=[train_len, val_len, test_len]
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
