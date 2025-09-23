

from torch.utils.data import Dataset,Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets


class GenericDatasetClass(Dataset):
    def __init__(self,transform=None,val_transform=None):
        self.transform = transform
        self.val_transform = val_transform if val_transform is not None else transform

    def get_dataloaders(self,batch_size=64):
        # Example using CIFAR10, replace with your dataset
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        train_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader