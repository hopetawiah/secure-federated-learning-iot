# ACES-FL Implementation Environment

The uploaded implementation-configuration document records the following environment:

| Component | Specification |
|---|---|
| Operating environment | Ubuntu under WSL on Windows |
| Programming language | Python |
| Federated-learning framework | Flower 1.31.0 |
| Machine-learning framework | PyTorch 2.10.0 |
| Simulation/runtime | Ray 2.55.1 |
| Dataset | UNSW-NB15 |
| Training records | 175,341 |
| Testing records | 82,332 |
| Input features after preprocessing | 194 |
| Simulated clients | 20 |
| Clients selected in ACES-FL rounds | 10 |
| Global training rounds | 20 |
| Model architecture | 194 -> 128 -> 64 -> 2 |
| Trainable parameters | 33,346 |
| Activation | ReLU |
| Dropout | 0.20 |
| Compression | FP32, FP16 and INT8 depending on client bandwidth |
| Baseline | FedAvg |
| Attack tested | Sign-flip model poisoning |
| Defense | robust norm deviation + cosine similarity + recoverable historical trust |
