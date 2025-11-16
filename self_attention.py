import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return (
            torch.tensor(self.X.iloc[ix].values).float(),
            torch.tensor(self.y.iloc[ix]).long(),
        )


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 64, Dataset=Dataset):
        super().__init__()
        self.batch_size = batch_size
        self.Dataset = Dataset

    def setup(self, stage=None):
        mnist = fetch_openml("mnist_784", version=1)
        X, y = mnist["data"], mnist["target"]
        X_train, X_test, y_train, y_test = (
            X[:60000] / 255.0,
            X[60000:] / 255.0,
            y[:60000].astype(np.int64),
            y[60000:].astype(np.int64),
        )
        self.train_ds = self.Dataset(X_train, y_train)
        self.val_ds = self.Dataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


dm = MNISTDataModule()
dm.setup()
batch = next(iter(dm.train_dataloader()))
imgs, labels = batch
print(imgs.shape, labels.shape)

r, c = 8, 8
fig = plt.figure(figsize=(c * 2, r * 2))
for _r in range(r):
    for _c in range(c):
        ix = _r * c + _c
        ax = plt.subplot(r, c, ix + 1)
        img, label = imgs[ix], labels[ix]
        ax.axis("off")
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(label.item())
plt.tight_layout()
plt.show()


class MLP(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(784, 784),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(784, 10),
        )

    def forward(self, x):
        return self.mlp(x)

    def predict(self, x):
        with torch.no_grad():
            y_hat = self(x)
            return torch.argmax(y_hat, axis=1)

    def compute_loss_and_acc(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, axis=1) == y).sum().item() / y.shape[0]
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log("loss", loss)
        self.log("acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        return optimizer


mlp = MLP()
outuput = mlp(torch.randn(64, 784))
print(outuput.shape)

mlp = MLP()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(mlp, dm)

imgs, labels = next(iter(dm.val_dataloader()))
preds = mlp.predict(imgs)

r, c = 8, 8
fig = plt.figure(figsize=(c * 2, r * 2))
for _r in range(r):
    for _c in range(c):
        ix = _r * c + _c
        ax = plt.subplot(r, c, ix + 1)
        img, label = imgs[ix], labels[ix]
        ax.axis("off")
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(
            f"{label.item()}/{preds[ix].item()}",
            color="green" if label == preds[ix] else "red",
        )
plt.tight_layout()
plt.show()


class AttnDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, patch_size=(7, 7)):
        self.X = X
        self.y = y
        self.patch_size = patch_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        image = torch.tensor(self.X.iloc[ix]).float().view(28, 28)  # 28 x 28
        h, w = self.patch_size
        patches = image.unfold(0, h, h).unfold(1, w, w)  # 4 x 4 x 7 x 7
        patches = patches.contiguous().view(-1, h * w)  # 16 x 49
        return patches, torch.tensor(self.y.iloc[ix]).long()


attn_dm = MNISTDataModule(Dataset=AttnDataset)
attn_dm.setup()
imgs, labels = next(iter(attn_dm.train_dataloader()))
print(img.shape, labels.shape)

fig = plt.figure(figsize=(5, 5))
for i in range(4):
    for j in range(4):
        ax = plt.subplot(4, 4, i * 4 + j + 1)
        ax.imshow(imgs[6, i * 4 + j].view(7, 7), cmap="gray")
        ax.axis("off")
plt.tight_layout()
plt.show()


class ScaledDotSelfAttention(torch.nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        # key, query, value projections
        self.key = torch.nn.Linear(n_embd, n_embd)
        self.query = torch.nn.Linear(n_embd, n_embd)
        self.value = torch.nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        # calculate query, key, values
        k = self.key(x)  # (B, L, F)
        q = self.query(x)  # (B, L, F)
        v = self.value(x)  # (B, L, F)

        # attention (B, L, F) x (B, F, L) -> (B, L, L)
        att = (q @ k.transpose(1, 2)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v  # (B, L, L) x (B, L, F) -> (B, L, F)

        return y


class Model(MLP):

    def __init__(self, n_embd=7 * 7, seq_len=4 * 4):
        super().__init__()
        self.mlp = None

        self.attn = ScaledDotSelfAttention(n_embd)
        self.actn = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(n_embd * seq_len, 10)

    def forward(self, x):
        x = self.attn(x)
        # print(x.shape)
        y = self.fc(self.actn(x.view(x.size(0), -1)))
        # print(y.shape)
        return y


model = Model()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, attn_dm)

attn_imgs, attn_labels = next(iter(attn_dm.val_dataloader()))
preds = model.predict(attn_imgs)

ix = random.randint(0, dm.batch_size)
fig = plt.figure(figsize=(5, 5))
for i in range(4):
    for j in range(4):
        ax = plt.subplot(4, 4, i * 4 + j + 1)
        ax.imshow(attn_imgs[ix, i * 4 + j].view(7, 7), cmap="gray")
        ax.axis("off")
fig.suptitle(
    f"{attn_labels[ix]} / {preds[ix].item()}",
    color="green" if attn_labels[ix] == preds[ix].item() else "red",
)
plt.tight_layout()
plt.show()
