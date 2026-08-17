import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42
NUM_CLIENTS = 20

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

CLIENT_DIR = "data/clients"
PROCESSED_DIR = "data/processed"

os.makedirs(CLIENT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 70)
print("ACES-FL UNSW-NB15 DATA PREPARATION")
print("=" * 70)

# Load datasets
print("\n[1/7] Loading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)

# Labels
print("\n[2/7] Separating features and labels...")

y_train = train_df["label"].astype(np.int64).values
y_test = test_df["label"].astype(np.int64).values

# Remove non-predictor columns
DROP_COLUMNS = ["id", "attack_cat", "label"]

X_train_df = train_df.drop(columns=DROP_COLUMNS)
X_test_df = test_df.drop(columns=DROP_COLUMNS)

print("Predictor columns:", X_train_df.shape[1])

# Feature types
categorical_columns = ["proto", "service", "state"]

numerical_columns = [
    col for col in X_train_df.columns
    if col not in categorical_columns
]

print("Categorical features:", categorical_columns)
print("Numerical features  :", len(numerical_columns))

# Preprocessing
print("\n[3/7] Fitting preprocessing pipeline...")


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_columns,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
            categorical_columns,
        ),
    ]
)

# Fit ONLY on training data
X_train = preprocessor.fit_transform(X_train_df)

# Use same transformation on test data
X_test = preprocessor.transform(X_test_df)

X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)

print("Processed training shape:", X_train.shape)
print("Processed testing shape :", X_test.shape)

# Save preprocessing objects

print("\n[4/7] Saving preprocessing information...")

joblib.dump(
    preprocessor,
    os.path.join(PROCESSED_DIR, "preprocessor.joblib")
)

np.savez_compressed(
    os.path.join(PROCESSED_DIR, "test_data.npz"),
    X=X_test,
    y=y_test,
)

feature_names = preprocessor.get_feature_names_out()

with open(
    os.path.join(PROCESSED_DIR, "feature_names.txt"),
    "w",
    encoding="utf-8",
) as f:
    for feature in feature_names:
        f.write(str(feature) + "\n")

print("Final number of model input features:", len(feature_names))

# Create 20 clients
print("\n[5/7] Creating 20 federated clients...")

skf = StratifiedKFold(
    n_splits=NUM_CLIENTS,
    shuffle=True,
    random_state=RANDOM_SEED,

)

client_metadata = []

for client_id, (_, client_indices) in enumerate(
    skf.split(X_train, y_train)
):
    client_X = X_train[client_indices]
    client_y = y_train[client_indices]

    client_path = os.path.join(
        CLIENT_DIR,
        f"client_{client_id:02d}.npz"
    )

    np.savez_compressed(
        client_path,
        X=client_X,
        y=client_y,
    )

    normal_count = int(np.sum(client_y == 0))
    attack_count = int(np.sum(client_y == 1))

    metadata = {
        "client_id": client_id,
        "records": int(len(client_y)),
        "normal": normal_count,
        "attack": attack_count,
        "normal_percent": round(
            normal_count / len(client_y) * 100, 2
        ),

        "attack_percent": round(
            attack_count / len(client_y) * 100, 2
        ),
    }

    client_metadata.append(metadata)

    print(
        f"Client {client_id:02d}: "
        f"{len(client_y):5d} records | "
        f"Normal={normal_count:4d} | "
        f"Attack={attack_count:4d}"
    )

# Save client metadata
metadata_df = pd.DataFrame(client_metadata)

metadata_df.to_csv(
    os.path.join(PROCESSED_DIR, "client_metadata.csv"),
    index=False,
)

# Save summary
with open(
    os.path.join(PROCESSED_DIR, "dataset_summary.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "training_records": int(len(y_train)),
            "testing_records": int(len(y_test)),

            "num_clients": NUM_CLIENTS,
            "input_features": int(X_train.shape[1]),
            "training_normal": int(np.sum(y_train == 0)),
            "training_attack": int(np.sum(y_train == 1)),
            "testing_normal": int(np.sum(y_test == 0)),
            "testing_attack": int(np.sum(y_test == 1)),
            "random_seed": RANDOM_SEED,
        },
        f,
        indent=4,
    )

# Verification
print("\n[6/7] Verifying partitions...")

total_client_records = metadata_df["records"].sum()

print("Original training records:", len(y_train))
print(
    "Records distributed across clients:",
    total_client_records
)

if total_client_records != len(y_train):
    raise ValueError(
        "Client partitions do not contain all training records."
    )

print("All training records successfully allocated.")

# Final summary
print("\n[7/7] DATA PREPARATION COMPLETE")

print("=" * 70)

print(f"Training records : {len(y_train):,}")
print(f"Testing records  : {len(y_test):,}")
print(f"Federated clients: {NUM_CLIENTS}")
print(f"Input dimensions : {X_train.shape[1]}")
print(f"Normal training  : {np.sum(y_train == 0):,}")
print(f"Attack training  : {np.sum(y_train == 1):,}")

print("=" * 70)
