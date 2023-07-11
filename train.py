import numpy as np
import torch
import wandb
import torch.nn as nn
import time
import covidx
from models import Classifier, Discriminator, get_densenet


def start_logging(model, epochs, start_time, model_name, train_batch_size, run_name):
    wandb.login(key="52a8fcda8d59e5f549aebc7b19c0689466f0cc0f")
    wandb.init(project=f"DA",
               config={"epochs": epochs,
                       "model": model_name,
                       "batch_size": train_batch_size})

    wandb.run.name = f"{start_time}_{run_name}"
    wandb.watch(model)


# Train the model
def train(device, dataloader, model, classifier, loss_fn, optimizer):
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
def test(dataloader, model, device, classifier, loss_fn):
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


def source_only(model, device, train_dl, test_dl, classifier, loss_fn, optimiser, epochs, patience, start_time):
    best_loss = np.Inf

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(device, train_dl, model, classifier, loss_fn, optimiser)
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


def eval_model(model, device, test_dl):
    classes = ["covid", "normal", "pneumonia"]

    model.eval()
    x, y = test_dl[0][0], test_dl[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    start_time = time.strftime('%Y%m%d_%H%M')

    train_dl, test_dl = covidx.get_source_covidx()

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else
              "cpu")

    print(f"Using {device} device")

    # Define model
    model = get_densenet().to(device)
    classifier = Classifier().to(device)
    print(model)

    # Optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)

    epochs = 100
    patience = 10

    start_logging(model, epochs, start_time, "DenseNet121", "64", "covidx")

    source_only(model, device, train_dl, test_dl, classifier, loss_fn, optimizer, epochs, patience, start_time)
    eval_model(model, device, test_dl)


if __name__ == '__main__':
    main()
