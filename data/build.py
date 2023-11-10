from torch.utils.data import DataLoader
from dataset import Food101


def build_dataset():
    train_dataset = Food101(mode = "train")
    test_dataset = Food101(mode = "test")
    train_dataloader = DataLoader(train_dataset, batch_size = 8, 
                                  shuffle= True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size = 8)
    return train_dataloader, test_dataloader
