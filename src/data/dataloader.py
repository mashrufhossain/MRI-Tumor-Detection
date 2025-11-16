from torch.utils.data import DataLoader
from torchvision import datasets
from src import config

def get_loaders():
    train_data = datasets.ImageFolder(
        root=config.TRAIN_DIR, transform=config.transform_train
    )
    test_data = datasets.ImageFolder(
        root=config.TEST_DIR, transform=config.transform_test
    )

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_data.classes
