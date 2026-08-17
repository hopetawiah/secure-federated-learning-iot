import csv
import json
from pathlib import Path

import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
)

from flwr.serverapp import (
    Grid,
    ServerApp,
)

from acesfldefense.resource_strategy import ResourceAwareFedAvg


from acesfldefense.task import (
    Net,
    evaluate_model,
    load_global_test_data,
    model_payload_bytes,
)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    """Run ACES-FL adaptive selection and compression on UNSW-NB15."""

    num_rounds = int(
        context.run_config["num-server-rounds"]
    )

    learning_rate = float(
        context.run_config["learning-rate"]
    )

    input_dim = int(
        context.run_config["input-dim"]
    )

    batch_size = int(
        context.run_config["batch-size"]
    )

    num_clients = int(

        context.run_config["num-clients"]
    )

    clients_per_round = int(
        context.run_config.get("clients-per-round", 10)
    )

    fairness_slots = int(
        context.run_config.get("fairness-slots", 2)
    )

    compression_low_threshold_mbps = float(
        context.run_config.get(
            "compression-low-threshold-mbps",
            10.0,
        )
    )

    compression_high_threshold_mbps = float(
        context.run_config.get(
            "compression-high-threshold-mbps",
            30.0,
        )
    )

    data_dir = context.run_config["data-dir"]

    results_dir = Path(
        context.run_config["results-dir"]
    )

    results_dir.mkdir(

        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(42)

    global_model = Net(
        input_dim=input_dim
    )

    arrays = ArrayRecord(
        global_model.state_dict()
    )

    payload_bytes = model_payload_bytes(
        global_model
    )

    print("=" * 70)
    print("ACES-FL SECURITY: ATTACK WITH ACES DEFENSE")
    print("=" * 70)

    print(
        f"Clients             : {num_clients}"
    )

    print(
        f"Model input features: {input_dim}"
    )

    print(
        f"Model payload       : "

        f"{payload_bytes:,} bytes"
    )

    print(
        f"Selected-client FP32 reference/round: "
        f"{payload_bytes * clients_per_round:,} bytes"
    )

    metrics_file = (
        results_dir
        / "aces_defense_metrics.csv"
    )

    compression_history_file = (
        results_dir
        / "compression_history.csv"
    )

    def read_round_communication(server_round):
        """Read actual communication evidence written by aggregate_train."""

        result = {
            "round_upload_bytes": 0,
            "flower_upload_bytes": 0,
            "successful_clients": 0,
            "fp32_clients": 0,
            "fp16_clients": 0,
            "int8_clients": 0,
        }

        # Round 0 is initial evaluation only.
        if int(server_round) == 0:

            return result

        if not compression_history_file.exists():
            return result

        rows = []

        with open(
            compression_history_file,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if int(row["round"]) == int(server_round):
                    rows.append(row)

        if not rows:
            return result

        result["round_upload_bytes"] = sum(
            int(row["compressed_array_bytes"])
            for row in rows
        )

        result["flower_upload_bytes"] = sum(
            int(row["flower_compression_payload_bytes"])
            for row in rows
        )


        result["successful_clients"] = len(rows)

        result["fp32_clients"] = sum(
            1
            for row in rows
            if row["compression_mode"].lower() == "fp32"
        )

        result["fp16_clients"] = sum(
            1
            for row in rows
            if row["compression_mode"].lower() == "fp16"
        )

        result["int8_clients"] = sum(
            1
            for row in rows
            if row["compression_mode"].lower() == "int8"
        )

        return result

    testloader = load_global_test_data(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    def global_evaluate(
        server_round,
        arrays,
    ):
        """Evaluate global model against untouched test data."""


        model = Net(
            input_dim=input_dim
        )

        model.load_state_dict(
            arrays.to_torch_state_dict()
        )

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        metrics = evaluate_model(
            model=model,
            dataloader=testloader,
            device=device,
        )

        print(
            f"\nGLOBAL ROUND {server_round}: "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

        communication = read_round_communication(
            server_round
        )


        file_exists = metrics_file.exists()

        with open(
            metrics_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "round",
                    "loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "model_payload_bytes",
                    "selected_fp32_reference_bytes",
                    "round_upload_bytes",
                    "flower_upload_bytes",
                    "successful_clients",
                    "fp32_clients",
                    "fp16_clients",
                    "int8_clients",
                ],
            )

            if not file_exists:
                writer.writeheader()


            writer.writerow(
                {
                    "round": server_round,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "model_payload_bytes":
                        payload_bytes,

                    "selected_fp32_reference_bytes":
                        (
                            payload_bytes
                            * communication[
                                "successful_clients"
                            ]
                        ),

                    "round_upload_bytes":
                        communication[
                            "round_upload_bytes"
                        ],

                    "flower_upload_bytes":
                        communication[
                            "flower_upload_bytes"
                        ],

                    "successful_clients":
                        communication[

                            "successful_clients"
                        ],

                    "fp32_clients":
                        communication[
                            "fp32_clients"
                        ],

                    "fp16_clients":
                        communication[
                            "fp16_clients"
                        ],

                    "int8_clients":
                        communication[
                            "int8_clients"
                        ],
                }
            )

        return MetricRecord(
            {
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    # Standard baseline:
    # all 20 clients participate.

    strategy = ResourceAwareFedAvg(
        clients_per_round=clients_per_round,
        fairness_slots=fairness_slots,
        compression_low_threshold_mbps=(
            compression_low_threshold_mbps
        ),
        compression_high_threshold_mbps=(
            compression_high_threshold_mbps
        ),
        results_dir=str(results_dir),
        seed=42,
        fraction_train=1.0,
        min_train_nodes=clients_per_round,
        min_available_nodes=num_clients,
        fraction_evaluate=0.0,
        min_evaluate_nodes=0,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr":
                    learning_rate,

                "malicious-clients-per-round":
                    int(
                        context.run_config.get(
                            "malicious-clients-per-round",
                            2,
                        )

                    ),

                "attack-type":
                    str(
                        context.run_config.get(
                            "attack-type",
                            "sign_flip",
                        )
                    ),

                "attack-scale":
                    float(
                        context.run_config.get(
                            "attack-scale",
                            5.0,
                        )
                    ),

                "attack-seed":
                    int(
                        context.run_config.get(
                            "attack-seed",
                            4242,
                        )
                    ),
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final global model

    final_state = (
        result.arrays
        .to_torch_state_dict()
    )

    final_path = (
        results_dir
        / "aces_defense_final_model.pt"
    )

    torch.save(
        final_state,
        final_path,
    )

    configuration = {
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "fairness_slots": fairness_slots,
        "selection_method": "ACES-Resource-Fairness",
        "num_rounds": num_rounds,
        "input_dim": input_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_payload_bytes":
            payload_bytes,

        "selected_fp32_reference_upload_per_round":
            payload_bytes * clients_per_round,

        "communication_metric":
            "client-to-server model/update tensor bytes",


        "flower_payload_metric":
            "serialized Flower ArrayRecord plus compression metadata",

        "compression_history_file":
            "compression_history.csv",
    }

    with open(
        results_dir / "aces_defense_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
        )

    print(
        "\nFinal model saved to:",
        final_path,
    )

    print(
        "Metrics saved to:",
        metrics_file,
    )
