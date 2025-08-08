from typing import final, override

import numpy as np
import optuna
import torch
import torch.nn as nn
from helper import benefit_of_doubt_acc
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import normalized_mutual_info_score
from torchvision.utils import save_image


def best_map(true, pred):
    true_labels = np.unique(true)
    pred_labels = np.unique(pred)
    n_labels = np.maximum(len(true_labels), len(pred_labels))
    G = np.zeros((n_labels, n_labels))
    for i in range(len(true_labels)):
        ind_cla1 = true == true_labels[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(len(pred_labels)):
            ind_cla2 = pred == pred_labels[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    new_pred_labels = np.zeros(pred.shape)
    for i in range(len(pred_labels)):
        print(true_labels, pred_labels)
        new_pred_labels[pred == pred_labels[i]] = true_labels[c[i]]
    return new_pred_labels


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def imgnorm(imgdata: np.array, low: float, high: float):
    from_low, from_high = np.min(imgdata), np.max(imgdata)
    return (imgdata - from_low) * (high - low) / (from_high - from_low) + low


def log_training(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    epoch: int,
    prefix: str,
    suffix: str = "",
):
    assert prefix.strip()  # prefix not blank
    import os

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    os.makedirs("training_log", exist_ok=True)
    filepath_no_ext = f"training_log/{TIMESTAMP}-{prefix}"

    with torch.no_grad():
        embeddings = model.encoder(X)
        # targets_label = np.argmax(all_p.cpu().numpy(), axis=1)
        # missrate_xkl = err_rate(y, targets_label)
        # acc = 1 - missrate_xkl
        q = 1.0 / (1.0 + torch.cdist(embeddings, model.centroids) ** 2)
        y_pred = torch.argmax(q, dim=1).cpu().numpy()
        acc = benefit_of_doubt_acc(y, y_pred)
        nmi = normalized_mutual_info_score(y, y_pred)

        recon, _ = model(X[:5])
        save_image(X[:5], f"{filepath_no_ext}_true.png")
        save_image(recon, f"{filepath_no_ext}_pred.png")

        tsne = TSNE(n_components=2).fit_transform(embeddings)
        plt.clf()
        plt.scatter(tsne[:, 0], tsne[:, 1], c=y, cmap="tab10", s=5)
        plt.colorbar()
        plt.savefig(f"{filepath_no_ext}_tsne.png")

        log = f"{prefix}: epoch: {epoch + 1} acc: {acc:.4f} nmi: {nmi:.4f} | {suffix}"
        with open(f"{filepath_no_ext}.log", "a") as f:
            f.write(log + "\n")
        print(log)

    return acc, nmi


@final
class Datasets:
    np.random.seed(42)

    @staticmethod
    def UNIPEN():
        from os import listdir
        from os.path import isfile, join

        import cv2

        X = []
        y = []
        for i in map(ord, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            path_img = f"/mnt/storage/datasets/UNIPEN-64x64-grayscale/{i}/"
            for file_name in [
                f for f in listdir(path_img) if isfile(join(path_img, f))
            ]:
                img = cv2.imread(path_img + file_name, 0)
                img = img.reshape(-1)
                X.append(img)
                y.append(i)

        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y, 52

    @staticmethod
    def MNIST():
        X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        return X, y, 10


@final
class Model(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        alpha: float = 1.0,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma

        # Build encoder
        self.encoder = nn.Sequential(
            # [32, 14, 14]
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.01),
            # [64, 7, 7]
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
        )

        # Build decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            # [32, 14, 14]
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.01),
            # [1, 28, 28]
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # output is in [0,1]
        )

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Trainable cluster centroids
        self.centroids = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(n_clusters, latent_dim), std=0.04)
        )

    @override
    def forward(self, x: torch.Tensor):
        # Encode input to lower-dimensional space
        embeddings = self.encoder(x)

        # Reconstruct input from embedding
        recon = self.decoder(embeddings)

        return recon, embeddings

    def cluster_loss(self, embeddings: torch.Tensor):
        dist = torch.cdist(embeddings, self.centroids) ** 2

        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)

        with torch.no_grad():
            p = q**2 / q.sum(dim=0)
            p = p / p.sum(dim=1, keepdim=True)

        loss = torch.mean(torch.sum(p * torch.log(1e-8 + p / (q + 1e-8)), dim=1))
        return loss, p

    def loss(self, x, recon, embeddings: torch.Tensor):
        # Focal MSE (downright small errors, common in sparse data)
        mse_loss = (x - recon) ** 2
        focal_weight = torch.abs(x - recon) ** self.gamma
        focal_loss = focal_weight * mse_loss
        recon_loss = torch.mean(focal_loss)

        # Clustering loss
        cluster_loss, p = self.cluster_loss(embeddings)

        # Combined loss with weighting coefficients
        loss = recon_loss + self.alpha * cluster_loss
        return loss, recon_loss, cluster_loss


def objective(trial: optuna.Trial):
    # Set TIMESTAMP for trial
    from datetime import datetime

    global TIMESTAMP
    TIMESTAMP = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    # Setup/optimize hyperparameters
    batch_size = trial.suggest_int("batch_size", 8, 128, step=8)
    latent_dim = trial.suggest_int("latent_dim", 5, 50)
    alpha = trial.suggest_float("alpha", 0.1, 2.0)
    pretraining_period = trial.suggest_float("pretraining_period", 0.3, 1.0)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = 65

    # Load dataset
    X, y, n_clusters = Datasets.MNIST()
    subset_len = 5000
    X = X[:subset_len]
    X = X.reshape(subset_len, 1, 28, 28)  # conv2d
    X = torch.FloatTensor(imgnorm(X, 0, 1))  # normalize
    y = np.array(y[:subset_len]).astype(float)
    dataloader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)

    # Create model
    model = Model(n_clusters, latent_dim, alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Pretrain autoencoder
    for epoch in range(int(epochs * pretraining_period)):
        for batch in dataloader:
            recon, _ = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if torch.isnan(loss):
            raise optuna.TrialPruned()

        if (epoch + 1) % 5 == 0:
            msg = f"loss: {loss.item():.4f}"
            acc, nmi = log_training(model, X, y, epoch, "AUTOENC", msg)

    # Initialize centroids
    with torch.no_grad():
        embeddings = model.encoder(X)
        kmeans = KMeans(n_clusters=n_clusters, n_init=100)
        kmeans.fit(embeddings)
        model.centroids.data.copy_(torch.tensor(kmeans.cluster_centers_))
        print("Centroids initialized with k-means")

    tol = 1e-3
    best_loss = torch.inf
    patience = 10
    no_improvement = 0

    # Joint training
    for epoch in range(epochs):
        for batch in dataloader:
            recon, embeddings = model(batch)
            loss, recon_loss, cluster_loss = model.loss(batch, recon, embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if torch.isnan(loss):
            raise optuna.TrialPruned()

        if (epoch + 1) % 5 == 0:
            msg = f"loss: {loss.item():.4f} recon_loss: {recon_loss.item():.4f} cluster_loss: {cluster_loss.item():.4f}"
            acc, nmi = log_training(model, X, y, epoch, "FITTING", msg)

        if loss < best_loss - tol:
            best_loss = loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == patience:
            return loss, acc, nmi

        # if epoch > 20:
        # Gradually decrease alpha
        # final_alpha = 0.1
        # progress = epoch / epochs
        # model.alpha = final_alpha + (alpha - final_alpha) * (1 - progress**0.5)

        # Gradually increase alpha
        # model.alpha = alpha * (1.0 + epoch / epochs)

    return loss, acc, nmi


def main():
    study = optuna.create_study(
        directions=["minimize", "maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.enqueue_trial(
        {
            "batch_size": 8,
            "latent_dim": 8,
            "lr": 0.00183,
            "alpha": 1.05,
            "pretraining_period": 0.6,
        }
    )
    study.optimize(objective, timeout=8 * 3600, n_trials=1)

    pareto_front = study.best_trials
    print(f"Number of Pareto-optimal trials: {len(pareto_front)}")

    for i, trial in enumerate(pareto_front):
        print(f"\nPareto Trial #{i + 1}:")
        print(f"  Loss: {trial.values[0]:.4f} (minimize)")
        print(f"  Accuracy: {trial.values[1]:.4f} (maximize)")
        print(f"  NMI: {trial.values[2]:.4f} (maximize)")
        print("  Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
