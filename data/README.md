# Dataset Documentation

## Dataset

This research uses the **UNSW-NB15 network intrusion detection dataset** for the experimental evaluation of the ACES-FL framework.

The official dataset files used in the study contained:

- **Training set:** 175,341 records
- **Testing set:** 82,332 records
- **Task:** Binary network intrusion classification
- **Federated clients:** 20 simulated clients

The dataset provides labelled normal and attack network traffic suitable for evaluating federated intrusion-detection systems.

---

## Data Preprocessing

Before federated training, the following preprocessing steps were performed:

1. Identifier and attack-category fields not required as model inputs were removed.
2. Categorical features were encoded, including:
   - protocol
   - service
   - state
3. Categorical variables were converted using one-hot encoding.
4. Numerical variables were standardised.
5. The final processed dataset contained **194 input features**.
6. The prediction target was converted to a binary label:
   - Normal traffic
   - Attack traffic

---

## Class Distribution

### Training Dataset

| Class | Records |
|---|---:|
| Attack | 119,341 |
| Normal | 56,000 |
| **Total** | **175,341** |

### Test Dataset

| Class | Records |
|---|---:|
| Attack | 45,332 |
| Normal | 37,000 |
| **Total** | **82,332** |

---

## Federated Data Partitioning

The preprocessed training dataset was divided across **20 stratified simulated federated clients**.

Each client contained approximately:

**8,767–8,768 training observations**

Stratified partitioning was used so that the simulated clients maintained meaningful representation of the binary normal/attack classes.

Raw training records remained within their simulated client partitions during federated training.

---

## Client Resource Profiles

The UNSW-NB15 dataset itself does not contain the resource conditions used by ACES-FL.

Separate simulated metadata was therefore created for each federated client to represent heterogeneous IoT conditions, including:

- Battery level
- CPU capability
- Network bandwidth
- Connectivity condition
- Participation history

These values support ACES-FL client-selection and communication-compression decisions.

They are **not included as intrusion-detection model features**.

---

## Communication Precision

Client bandwidth determines the model-update representation used by ACES-FL:

| Simulated Bandwidth | Update Representation |
|---|---|
| ≥ 30 Mbps | FP32 |
| 10 to < 30 Mbps | FP16 |
| < 10 Mbps | INT8 |

---

## Data Usage

The dataset is used to evaluate:

- Federated intrusion detection
- Predictive accuracy
- Precision
- Recall
- F1-score
- Federated convergence
- Communication efficiency
- Client participation fairness
- Model-poisoning resilience
- Malicious-update detection

---

## Reproducibility

Dataset preprocessing and partitioning scripts will be maintained separately from the raw dataset.

Researchers reproducing this work should obtain the **UNSW-NB15 dataset from its authorised/public source** and follow the preprocessing procedure documented in this repository.

The raw dataset is not redistributed through this repository.

---

## Experimental Scope

The UNSW-NB15 dataset provides a controlled benchmark for evaluating ACES-FL, but the experiment should not be interpreted as a complete physical IoT deployment.

The dissertation identifies future validation using additional IoT-focused datasets and physical or emulated resource-constrained devices.
