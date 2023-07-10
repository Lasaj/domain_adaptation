import torchvision
import torch
import wandb
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import DenseNet, DenseNet121_Weights

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                 transforms.Normalize(mean, std),
                                 ])

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms)
batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu")

print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT).to(device)
model.classifier = nn.Linear(1024, 10).to(device)
model.classifier = nn.Identity().to(device)

classifier = Classifier().to(device)


print(model)

# Optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)


def start_logging(model, epochs, model_name, train_batch_size, run_name):
    wandb.login(key="52a8fcda8d59e5f549aebc7b19c0689466f0cc0f")
    wandb.init(project=f"DA",
               config={"epochs": epochs,
                       "model": model_name,
                       "batch_size": train_batch_size})

    wandb.run.name = f"{time.strftime('%Y%m%d_%H%M')}_{run_name}"
    wandb.watch(model)


start_logging(model, "100", "DenseNet121", "64", "MNIST_ext_classifier")


# Train the model
def train(dataloader, model, classifier, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        feats = model(X)
        pred = classifier(feats)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            wandb.log({"training loss": loss})


# Test the model
def test(dataloader, model, classifier, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    classifier.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            feats = model(X)
            pred = classifier(feats)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    wandb.log({"test loss": test_loss, "accuracy": 100 * correct})


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, classifier, loss_fn, optimizer)
    test(test_dataloader, model, classifier, loss_fn)
print("Done!")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
