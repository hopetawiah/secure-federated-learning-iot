import time
import torch

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp

from acesflbaseline.task import (
    Net,
    load_client_data,
    model_payload_bytes,
    train_model,
)



app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the global model on one local UNSW-NB15 client."""

    partition_id = int(
        context.node_config["partition-id"]
    )

    data_dir = context.run_config["data-dir"]

    batch_size = int(
        context.run_config["batch-size"]
    )

    local_epochs = int(
        context.run_config["local-epochs"]
    )

    input_dim = int(
        context.run_config["input-dim"]
    )

    lr = float(
        msg.content["config"]["lr"]
    )

    # Initialise client model

    model = Net(input_dim=input_dim)

    # Load global model received from server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Load ONLY this client's local data
    trainloader = load_client_data(
        data_dir=data_dir,
        partition_id=partition_id,
        batch_size=batch_size,
    )

    start_time = time.time()

    train_loss = train_model(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
    )

    training_time = time.time() - start_time


    payload_bytes = model_payload_bytes(model)

    print(
        f"Client {partition_id:02d} | "
        f"Records={len(trainloader.dataset)} | "
        f"Loss={train_loss:.6f} | "
        f"Time={training_time:.2f}s | "
        f"Payload={payload_bytes:,} bytes"
    )

    model_record = ArrayRecord(
        model.state_dict()
    )

    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "training_time": float(training_time),
            "model_bytes": int(payload_bytes),
            "num-examples": int(
                len(trainloader.dataset)
            ),
        }
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metrics,
        }
    )


    return Message(
        content=content,
        reply_to=msg,
    )
