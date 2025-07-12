import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
import torchvision


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
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super().__init__()
        self.autoencoder = AutoEncoder(input_dim, hidden_dims, latent_dim)
        self.cluster_layer = ClusterLayer(n_clusters, latent_dim)

    def forward(self, x):
        z, x_rec = self.autoencoder(x)
        q = self.cluster_layer(z)
        return x_rec, z, q

    def predict(self, x):
        _, _, q = self.forward(x)
        return torch.argmax(q, dim=1)

    def fit(self, dataloader, lr=1e-3, epochs=100, alpha=0.7, beta=0.2, gamma=0.1):
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Pretrain autoencoder on 30% of the data
        for _ in range(int(epochs / 30)):
            for batch, _ in dataloader:
                # [batch_size, x, y, z] x [x, y, z, hidden_dim_size]
                x = batch.view(batch.shape[0], -1)
                optimizer.zero_grad()
                x_rec, _, _ = self(x)
                loss = criterion(x_rec, x)
                loss.backward()
                optimizer.step()

        # Initialize cluster centers with K-means
        embeddings = []
        with torch.no_grad():
            for batch, _ in dataloader:
                x = batch.view(batch.shape[0], -1)
                _, z, _ = self(x)
                embeddings.append(z)
        embeddings = torch.cat(embeddings, dim=0)
        kmeans = KMeans(n_clusters=self.cluster_layer.cluster_centers.shape[0])
        kmeans.fit(embeddings.numpy())
        self.cluster_layer.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        )

        # Joint training
        for epoch in range(epochs):
            for batch, _ in dataloader:
                x = batch.view(batch.shape[0], -1)
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

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def benefit_of_doubt_acc(y_true, y_pred):
    import numpy as np
    from sklearn import metrics
    from scipy.stats import mode

    most_common_label = mode(y_true)[0]
    label_mapping = {}
    for cluster in np.unique(y_pred):
        if cluster == -1:  # if treated as noise by density-based algorithms
            label_mapping[cluster] = most_common_label
        else:
            true_label = mode(y_true[y_pred == cluster])[0]
            label_mapping[cluster] = true_label

    aligned_labels = np.array(
        [label_mapping.get(pred, most_common_label) for pred in y_pred]
    )
    return metrics.accuracy_score(y_true, aligned_labels)


def train_and_save(dataset, save_filepath):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    clusterer = NDDC(
        input_dim=28 * 28, hidden_dims=[512, 256], latent_dim=128, n_clusters=10
    )
    clusterer.fit(dataloader, lr=1e-4, alpha=2.0, beta=0.05, gamma=0.05)
    torch.save(
        clusterer.state_dict(save_filepath),
    )


def load_and_calculate_accuracy(dataset, save_filepath):
    X, y_true = zip(*dataset)
    X = torch.stack(X)
    X = X.view(X.size(0), -1)
    y_true = torch.tensor(y_true)

    clusterer = NDDC(
        input_dim=28 * 28, hidden_dims=[512, 256], latent_dim=128, n_clusters=10
    )
    clusterer.load_state_dict(torch.load(save_filepath))
    y_pred = clusterer.predict(X)
    print(torch.unique(y_pred))

    print(benefit_of_doubt_acc(y_true, y_pred))


if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST(
        root="/home/piotr/PW/Inżynierka/data",
        transform=torchvision.transforms.ToTensor(),
    )
    dataset = torch.utils.data.Subset(dataset, range(5000))

    load_and_calculate_accuracy(dataset, "nddc-5000.pt")
    # train_and_save(dataset)
