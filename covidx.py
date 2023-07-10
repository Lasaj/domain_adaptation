import torch
import os
import shutil
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

data_dir = "./old_covid_x/"
source_dir = f"{data_dir}/source"
target_dir = f"{data_dir}/target"

target_domain_urls = [
    "https://github.com/armiro/COVID-CXNet",
    "https://eurorad.org",
    "https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png",
    "https://github.com/ieee8023/covid-chestxray-dataset",
    "https://sirm.org/category/senza-categoria/covid-19/",
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                 transforms.Normalize(mean, std),
                                 ])


def get_source_covidx():
    # Load data
    train_data = datasets.ImageFolder(root=source_dir, transform=transforms)

    # Split data
    train_size = int(0.85 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # train_size = int(len(train_data) - val_size)
    # train_data, test_data = random_split(train_data, [train_size, val_size])

    # Define dataloaders
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader  #, test_dataloader


def split_domains():
    covid_df = pd.read_excel(f"{data_dir}/COVID.metadata.xlsx")
    print(covid_df.URL.value_counts())
    # if folder does not exist, create it
    if not os.path.exists(f"{data_dir}/source"):
        os.makedirs(f"{data_dir}/source")
    if not os.path.exists(f"{data_dir}/target"):
        os.makedirs(f"{data_dir}/target/COVID")

    # split data into source and target domains
    target_df = covid_df[covid_df.URL.isin(target_domain_urls)]
    target_df.to_excel(f"{data_dir}/target/target_COVID.metadata.xlsx")
    print(target_df.columns)
    print(target_df.head)
    for f in target_df["FILE NAME"]:
        if os.path.exists(f"{data_dir}/COVID/{f}.png"):
            os.replace(f"{data_dir}/COVID/{f}.png", f"{data_dir}/target/COVID/{f}.png")

    source_df = covid_df[~covid_df.URL.isin(target_domain_urls)]
    source_df.to_excel(f"{data_dir}/source/source_COVID.metadata.xlsx")
    os.replace(f"{data_dir}/COVID", f"{data_dir}/source/COVID")
    os.replace(f"{data_dir}/Viral Pneumonia", f"{data_dir}/source/Viral Pneumonia")
    os.replace(f"{data_dir}/Normal", f"{data_dir}/source/Normal")
    shutil.copy(f"{data_dir}/Normal.metadata.xlsx", f"{data_dir}/source/Normal.metadata.xlsx")
    shutil.copy(f"{data_dir}/Viral Pneumonia.metadata.xlsx", f"{data_dir}/source/Viral Pneumonia.metadata.xlsx")


def main():
    # train, val, test = get_covidx()
    # print(len(train), len(val), len(test))
    # print(len(train.dataset), len(val.dataset), len(test.dataset))
    split_domains()


if __name__ == "__main__":
    main()
