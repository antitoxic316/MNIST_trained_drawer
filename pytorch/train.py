import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os

import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

from CNN import CNNmodel

SEED = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

train_transform = transforms.Compose([
    #transforms.TrivialAugmentWide(num_magnitude_bins=4),
    transforms.ToTensor()
])

test_transform = transforms.ToTensor()


train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

channel_num = train_data[0][0].shape[0]
model = CNNmodel(in_shape=channel_num, out_shape=len(train_data.classes)).to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 20
writer = SummaryWriter(log_dir="runs\\CNN_MNIST")

def train_step(dataloader, loss_fn, optimizer, model, device):
    train_loss = 0
    train_acc = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        torch.cuda.empty_cache()
        del X, y
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return (train_loss, train_acc)

def test_step(dataloader, loss_fn, model, device):
    test_loss = 0
    test_acc = 0

    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        test_loss += loss

        y_pred_class = torch.argmax(y_pred, dim=1)
        test_acc += (y_pred_class == y).sum().item()/len(y_pred)

        torch.cuda.empty_cache()
        del X, y
        
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return (test_loss, test_acc)


for epoch in range(epochs):
    train_loss, train_acc = train_step(
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        model=model,
        device=device
    )
    test_loss, test_acc = test_step(
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        model=model,
        device=device
    )

    writer.add_scalars(
        main_tag="Loss",
        tag_scalar_dict={"train_loss": train_loss,
                        "test_loss": test_loss   },
        global_step=epoch
    )

    writer.add_scalars(
        main_tag="Accuracy", 
        tag_scalar_dict={"train_acc": train_acc,
                        "test_acc": test_acc   }, 
        global_step=epoch
    )
    
    print(f"epoch={epoch}, train loss={train_loss}, train acc={train_acc}, test loss={test_loss}, test acc={test_acc}\n")
writer.close()

torch.save(model.state_dict(), f="CNN.pth")