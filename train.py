import numpy as np
import torchvision
import torch
import wandb
import torch.nn as nn
import time
import covidx
from torchvision.models import DenseNet, DenseNet121_Weights

start_time = time.strftime('%Y%m%d_%H%M')

train_dl, test_dl = covidx.get_source_covidx()

device = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu")

print(f"Using {device} device")


# Define model
class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT).to(device)
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

    wandb.run.name = f"{start_time}_{run_name}"
    wandb.watch(model)


start_logging(model, "100", "DenseNet121", "64", "covidx")


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
    return test_loss


epochs = 100
best_loss = np.Inf
patience = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dl, model, classifier, loss_fn, optimizer)
    test_loss = test(test_dl, model, classifier, loss_fn)
    # early stopping
    if test_loss < best_loss:
        best_loss = test_loss
        # save model weights
        torch.save(model.state_dict(), f"{start_time}_covidx_densenet_weights.pt")
    else:
        patience -= 1
        if patience == 0:
            break
print("Done!")

classes = ["covid", "normal", "pneumonia"]

model.eval()
x, y = test_dl[0][0], test_dl[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
