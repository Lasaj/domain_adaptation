import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

data_dir = "./old_covid_x/"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                 transforms.Normalize(mean, std),
                                 ])


def get_covidx():
    # Load data
    train_data = datasets.ImageFolder(root=data_dir, transform=transforms)

    # Split data
    train_size = int(0.85 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_size = int(len(train_data) - val_size)
    train_data, test_data = random_split(train_data, [train_size, val_size])

    # Define dataloaders
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def main():
    train, val, test = get_covidx()
    print(len(train), len(val), len(test))
    print(len(train.dataset), len(val.dataset), len(test.dataset))

if __name__ == "__main__":
    main()
