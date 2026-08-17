from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Net(nn.Module):
    """Binary intrusion-detection MLP for UNSW-NB15."""

    def __init__(self, input_dim: int = 194):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)



def load_client_data(
    data_dir: str,
    partition_id: int,
    batch_size: int,
):
    """Load one client's local UNSW-NB15 partition."""

    path = Path(data_dir) / "clients" / f"client_{partition_id:02d}.npz"

    data = np.load(path)

    x = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    dataset = TensorDataset(x, y)

    generator = torch.Generator()
    generator.manual_seed(42 + partition_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    return loader


def load_global_test_data(
    data_dir: str,
    batch_size: int,

):
    """Load untouched global UNSW-NB15 test dataset."""

    path = Path(data_dir) / "processed" / "test_data.npz"

    data = np.load(path)

    x = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def train_model(
    model,
    trainloader,
    epochs,
    lr,
    device,
):
    """Train one client's local model."""

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    total_loss = 0.0
    batches = 0

    for _ in range(epochs):

        for features, labels in trainloader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

    return total_loss / max(batches, 1)



def evaluate_model(
    model,
    dataloader,
    device,
):
    """Evaluate model using binary IDS metrics."""

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    batches = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for features, labels in dataloader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(
                outputs,
                dim=1,

            )

            total_loss += loss.item()
            batches += 1

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,

        all_predictions,
        zero_division=0,
    )

    avg_loss = total_loss / max(batches, 1)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def model_payload_bytes(model):
    """Calculate raw model parameter payload size."""

    total = 0

    for parameter in model.state_dict().values():

        total += (
            parameter.numel()
            * parameter.element_size()
        )

    return int(total)
