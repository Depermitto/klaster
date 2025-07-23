from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import fetch_openml
from munkres import Munkres
import torch
import torch.nn as nn
import numpy as np
import optuna


class CharCluster(nn.Module):
    def __init__(
        self,
        input_dim,
        architecture,
        latent_dim,
        n_clusters,
        alpha=1.0,
        beta=1.0,
        lam=1.0,
        hyper_m=1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.hyper_m = hyper_m

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in architecture:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(architecture):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

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

    def clustering_layer(self, embedding):
        # Calculate pairwise distances (shape: [batch_size, n_clusters, embedding_dim])
        diff = embedding.unsqueeze(1) - self.centroids

        # Compute probabilities (shape: [batch_size, n_clusters])
        exponent = 2.0 / (self.hyper_m - 1.0 + 1e-8)
        p = 1.0 / (torch.pow(diff, exponent).sum(dim=2) + 1e-8)

        # Normalize probabilities per sample
        p = p / p.sum(dim=1, keepdim=True)

        return p

    def forward(self, x):
        # Encode input to lower-dimensional space
        embedding = self.encoder(x)

        # Reconstruct input from embedding
        recon = self.decoder(embedding)

        # Compute cluster membership probabilities
        p = self.clustering_layer(embedding)

        return recon, embedding, p

    def build_cluster_loss(self, embedding, p):
        # Raise probabilities to power of m (fuzziness coefficient)
        p = torch.pow(p, self.hyper_m)

        # Calculate squared distances between embeddings and centroids
        distances = torch.sum((embedding.unsqueeze(1) - self.centroids) ** 2, dim=2)

        # Compute weighted average of distances (FCM objective)
        cluster_loss = torch.mean(torch.sum(p * distances, dim=1))

        return cluster_loss

    def build_loss(self, x, recon, embedding, p):
        # Reconstruction loss (MSE)
        recon_loss = torch.mean(torch.sum((recon - x) ** 2, dim=1))

        # Clustering loss
        cluster_loss = self.build_cluster_loss(embedding, p)

        # Entropy regularization terms
        f_j = torch.sum(p, dim=0)
        entropy = torch.sum(f_j * torch.log(f_j + 1e-8))
        con_entropy = -torch.sum(p * torch.log(p + 1e-8))

        # Combined loss with weighting coefficients
        loss = (
            recon_loss
            + self.alpha * cluster_loss
            + self.beta * con_entropy
            + self.lam * entropy
        )
        return loss, recon_loss, cluster_loss


def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def imgnorm(imgdata: np.array, low: float, high: float):
    from_low, from_high = np.min(imgdata), np.max(imgdata)
    return (imgdata - from_low) * (high - low) / (from_high - from_low) + low


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 64, 128, 256])
    architecture = [
        trial.suggest_int("layer1_units", 256, 2048, step=128),
        trial.suggest_int("layer2_units", 128, 1024, step=64),
        trial.suggest_int("layer3_units", 32, 512, step=32),
    ]
    latent_dim = trial.suggest_int("latent_dim", 5, 50)
    # alpha = trial.suggest_float("alpha", 0.1, 2.0)
    # beta = trial.suggest_float("beta", 0.1, 2.0)
    # lam = trial.suggest_float("lam", 0.1, 2.0)
    # hyper_m = trial.suggest_float("hyper_m", 0.5, 2.0)
    alpha = 1.0
    beta = 1.1
    lam = 1.0
    hyper_m = 1.5
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = 65

    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    subset_len = 20000
    X = X[:subset_len]
    X = torch.FloatTensor(imgnorm(X, -1, 1))
    y = np.array(y[:subset_len]).astype(float)
    dataloader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)

    model = CharCluster(
        input_dim=X.shape[1],
        architecture=architecture,
        latent_dim=latent_dim,
        n_clusters=10,
        alpha=alpha,
        beta=beta,
        lam=lam,
        hyper_m=hyper_m,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Pretrain autoencoder
    for epoch in range(int(epochs * 0.3)):
        for batch in dataloader:
            recon, _, _ = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                _, _, all_p = model(X)
                targets_label = np.argmax(all_p.cpu().numpy(), axis=1)
                missrate_xkl = err_rate(y, targets_label)
                acc = 1 - missrate_xkl
                nmi = normalized_mutual_info_score(y, targets_label)

                trial.report(nmi, epoch)
                print(
                    f"AUTOENC: epoch: {epoch + 1} acc: {acc:.4f} nmi: {nmi:.4f} | loss: {loss.item():.4f}"
                )

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    for epoch in range(epochs):
        for batch in dataloader:
            recon, embedding, p = model(batch)
            loss, recon_loss, cluster_loss = model.build_loss(
                batch, recon, embedding, p
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                _, _, all_p = model(X)
                targets_label = np.argmax(all_p.cpu().numpy(), axis=1)
                missrate_xkl = err_rate(y, targets_label)
                acc = 1 - missrate_xkl
                nmi = normalized_mutual_info_score(y, targets_label)

                trial.report(nmi, epochs + epoch)
                print(
                    f"FITTING: epoch: {epoch + 1} acc: {acc:.4f} nmi: {nmi:.4f} | loss: {loss.item():.4f} recon_loss: {recon_loss.item():.4f} cluster_loss: {cluster_loss.item():.4f}",
                )

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    return nmi


def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.enqueue_trial(
        {
            "batch_size": 128,
            "lr": 1e-3,
            "layer1_units": 1024,
            "layer2_units": 512,
            "layer3_units": 228,
            "latent_dim": 8,
            "alpha": 1.0,
            "beta": 1.1,
            "lam": 1.0,
            "hyper_m": 1.5,
        }
    )
    study.optimize(objective)

    print("Best trial:")
    trial = study.best_trial
    print(f"  NMI: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
