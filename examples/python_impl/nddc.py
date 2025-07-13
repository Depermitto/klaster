import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


class ClusterLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, z):
        # Student's t-distribution, soft assignment
        q = 1.0 / (
            1.0
            + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
            / self.alpha
        )
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q


def target_distribution(q):
    weight = q**2 / torch.sum(q, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


# N-Dimensional Deep Clusterer
class NDDC(nn.Module):
    def __init__(
        self, input_dim: int, architecture: list[int], latent_dim: int, n_clusters: int
    ):
        super().__init__()
        self.autoencoder = AutoEncoder(input_dim, architecture, latent_dim)
        self.cluster_layer = ClusterLayer(n_clusters, latent_dim)

    def forward(self, x: torch.Tensor):
        z, x_rec = self.autoencoder(x)
        q = self.cluster_layer(z)
        return x_rec, z, q

    def predict(self, x: torch.Tensor):
        _, _, q = self.forward(x)
        return torch.argmax(q, dim=1)

    def fit(
        self,
        X: torch.Tensor,
        model_id: str,
        lr: float = 1e-3,
        epochs: int = 100,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
    ):
        optimizer = Adam(self.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(X, batch_size=64, shuffle=True)
        criterion = nn.MSELoss()

        # Pretrain autoencoder on 30% of epochs
        for _ in range(int(epochs / 30)):
            for x in dataloader:
                optimizer.zero_grad()
                x_rec, _, _ = self(x)
                loss = criterion(x_rec, x)
                loss.backward()
                optimizer.step()

        # Initialize cluster centers with K-means
        embeddings = []
        with torch.no_grad():
            for x in dataloader:
                _, z, _ = self(x)
                embeddings.append(z)
        embeddings = torch.cat(embeddings, dim=0)
        kmeans = KMeans(n_clusters=self.cluster_layer.cluster_centers.shape[0])
        kmeans.fit(embeddings.numpy())
        self.cluster_layer.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        )

        # Joint training
        for _ in range(epochs):
            for x in dataloader:
                optimizer.zero_grad()
                x_rec, z, q = self(x)

                # Reconstruction loss
                rec_loss = F.mse_loss(x_rec, x)

                # Target distribution
                p = q**2 / torch.sum(q, dim=0)
                p = p / torch.sum(p, dim=1, keepdim=True)
                # Clustering loss (KL divergence between q and target distribution p)
                clustering_loss = F.kl_div(q.log(), p, reduction="batchmean")

                # Constrastive loss
                z = F.normalize(z, dim=1)
                temperature = 0.5
                similarity = torch.matmul(z, z.t()) / temperature
                labels = torch.arange(z.size(0)).to(z.device)
                constrastive_loss = F.cross_entropy(similarity, labels)

                loss = (
                    alpha * rec_loss
                    + beta * clustering_loss
                    + gamma * constrastive_loss
                )

                loss.backward()
                optimizer.step()

        torch.save(self.state_dict(), model_id)
