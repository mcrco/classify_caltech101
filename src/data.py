from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor
import lightning as L

class Caltech101DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, img_size, split_props):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.split_props = split_props
        self.transform = transforms.Compose([
            Resize(size=self.img_size),
            ToTensor()
        ])

    def prepare_data(self):
        datasets.Caltech101(root=self.data_dir, download=True)

    def setup(self, stage):
        data = datasets.Caltech101(
                root=self.data_dir, 
                download=False, 
                transform=self.transform
        )
        valid_indices = []
        for i in range(len(data)):
            if data[i][0].shape[0] == 3:
                valid_indices.append(i)
        data = Subset(data, valid_indices)

        self.train_data, self.test_data, self.val_data = random_split(data, self.split_props)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
