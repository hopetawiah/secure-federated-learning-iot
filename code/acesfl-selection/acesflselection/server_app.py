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

from acesflselection.resource_strategy import ResourceAwareFedAvg

from acesflselection.task import (
    Net,
    evaluate_model,
    load_global_test_data,
    model_payload_bytes,
)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    """Run ACES-FL resource-aware client selection on UNSW-NB15."""

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
    print("ACES-FL RESOURCE-AWARE CLIENT SELECTION")
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
        f"Upload/round approx.: "
        f"{payload_bytes * clients_per_round:,} bytes"
    )

    metrics_file = (
        results_dir
        / "fedavg_baseline_metrics.csv"
    )

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
                    "round_upload_bytes",
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
                    "model_payload_bytes": payload_bytes,
                    "round_upload_bytes":
                        payload_bytes * clients_per_round,
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
                "lr": learning_rate,
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
        / "fedavg_final_model.pt"
    )

    torch.save(
        final_state,
        final_path,
    )

    configuration = {
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "selection_method": "ACES-Resource-Fairness",
        "num_rounds": num_rounds,
        "input_dim": input_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_payload_bytes":
            payload_bytes,
        "estimated_upload_per_round":
            payload_bytes * clients_per_round,
    }

    with open(
        results_dir / "fedavg_config.json",
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
