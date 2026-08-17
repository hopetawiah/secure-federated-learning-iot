import time

import numpy as np
import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp

from acesflsecurity.attack import (
    delta_l2_norm,
    poison_delta,

)

from acesflsecurity.compression import (
    compress_update,
    decompress_update,
    raw_tensor_bytes,
    relative_l2_error,
)

from acesflsecurity.task import (
    Net,
    load_client_data,
    train_model,
)


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """
    Train one UNSW-NB15 client and upload a compressed
    MODEL UPDATE rather than the complete local model.
    """

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

    config = msg.content["config"]

    lr = float(
        config["lr"]
    )

    compression_mode = str(
        config["compression-mode"]
    )

    bandwidth_mbps = float(
        config["bandwidth-mbps"]
    )

    # -------------------------------------------------
    # Controlled security experiment configuration
    # -------------------------------------------------

    is_malicious = (
        int(

            config.get(
                "is-malicious",
                0,
            )
        )
        == 1
    )

    attack_type = str(
        config.get(
            "attack-type",
            "none",
        )
    )

    attack_scale = float(
        config.get(
            "attack-scale",
            1.0,
        )
    )

    server_round = int(
        config.get(
            "server-round",
            0,
        )
    )

    effective_attack_type = (
        attack_type
        if is_malicious

        else "none"
    )

    # -------------------------------------------------
    # Receive current global model
    # -------------------------------------------------

    global_state = (
        msg.content["arrays"]
        .to_torch_state_dict()
    )

    # Keep an immutable copy so that after local
    # training we can calculate:
    #
    # delta = local_model - global_model
    global_before = {
        key: value.detach().cpu().clone()
        for key, value in global_state.items()
    }

    model = Net(
        input_dim=input_dim
    )

    model.load_state_dict(
        global_state
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()

        else "cpu"
    )

    trainloader = load_client_data(
        data_dir=data_dir,
        partition_id=partition_id,
        batch_size=batch_size,
    )

    # -------------------------------------------------
    # Local training
    # -------------------------------------------------

    start_time = time.time()

    train_loss = train_model(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
    )

    training_time = (
        time.time() - start_time
    )

    # -------------------------------------------------
    # Calculate LOCAL MODEL UPDATE
    # -------------------------------------------------

    local_after = {

        key: value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }

    state_keys = list(
        global_before.keys()
    )

    delta = [
        (
            local_after[key]
            - global_before[key]
        )
        .numpy()
        .astype(np.float32)
        for key in state_keys
    ]

    # -------------------------------------------------
    # Controlled Byzantine model poisoning
    # -------------------------------------------------
    #
    # IMPORTANT:
    # The clean local update is measured first.
    # Poisoning occurs BEFORE adaptive compression.
    #
    # Benign:
    #     transmitted_delta = delta
    #
    # Malicious sign-flip:
    #     transmitted_delta = -attack_scale * delta

    # -------------------------------------------------

    clean_delta_norm = delta_l2_norm(
        delta
    )

    if is_malicious:
        delta = poison_delta(
            delta,
            attack_type=attack_type,
            attack_scale=attack_scale,
        )

    transmitted_delta_norm = (
        delta_l2_norm(
            delta
        )
    )

    if clean_delta_norm > 0:
        attack_amplification = (
            transmitted_delta_norm
            / clean_delta_norm
        )
    else:
        attack_amplification = 0.0

    raw_update_bytes = (
        raw_tensor_bytes(delta)
    )

    # -------------------------------------------------

    # Adaptive compression
    # -------------------------------------------------

    compressed_arrays, metadata = (
        compress_update(
            delta,
            compression_mode,
        )
    )

    compressed_record = ArrayRecord(
        compressed_arrays
    )

    compressed_array_bytes = (
        raw_tensor_bytes(
            compressed_arrays
        )
    )

    flower_array_bytes = int(
        compressed_record.count_bytes()
    )

    # Check distortion introduced by compression
    reconstructed = decompress_update(
        compressed_arrays,
        metadata,
    )

    reconstruction_error = (
        relative_l2_error(

            delta,
            reconstructed,
        )
    )

    # INT8 requires one scale per tensor.
    # FP16/FP32 require no scales.
    compression_info_dict = {
        "mode": compression_mode,
        "bandwidth-mbps": float(bandwidth_mbps),
        "partition-id": int(partition_id),
        "is-malicious": int(is_malicious),
        "attack-type": effective_attack_type,
        "attack-scale": (
            float(attack_scale)
            if is_malicious
            else 0.0
        ),
    }

    # INT8 needs scales; FP32 and FP16 do not.
    if metadata["scales"]:
        compression_info_dict["scales"] = [
            float(scale)
            for scale in metadata["scales"]
        ]

    compression_info = ConfigRecord(
        compression_info_dict
    )

    compression_metadata_bytes = int(

        compression_info.count_bytes()
    )

    flower_compression_payload = (
        flower_array_bytes
        + compression_metadata_bytes
    )

    reduction = (
        1.0
        - (
            compressed_array_bytes
            / raw_update_bytes
        )
    ) * 100.0

    print(
        f"Client {partition_id:02d} | "
        f"BW={bandwidth_mbps:.2f} Mbps | "
        f"Mode={compression_mode.upper()} | "
        f"Attack={effective_attack_type} | "
        f"Loss={train_loss:.6f} | "
        f"Raw={raw_update_bytes:,} B | "
        f"Compressed="
        f"{compressed_array_bytes:,} B | "
        f"Reduction={reduction:.2f}% | "
        f"Error="
        f"{reconstruction_error:.6f}"
    )

    metrics = MetricRecord(
        {

            "train_loss":
                float(train_loss),

            "training_time":
                float(training_time),

            "num-examples":
                int(len(trainloader.dataset)),

            "raw_update_bytes":
                int(raw_update_bytes),

            "compressed_array_bytes":
                int(compressed_array_bytes),

            "flower_array_bytes":
                int(flower_array_bytes),

            "compression_metadata_bytes":
                int(compression_metadata_bytes),

            "flower_compression_payload_bytes":
                int(flower_compression_payload),

            "compression_error":
                float(reconstruction_error),

            "is_malicious":
                int(is_malicious),

            "clean_delta_norm":
                float(clean_delta_norm),


            "transmitted_delta_norm":
                float(transmitted_delta_norm),

            "attack_amplification":
                float(attack_amplification),

            "attack_scale":
                (
                    float(attack_scale)
                    if is_malicious
                    else 0.0
                ),
        }
    )

    content = RecordDict(
        {
            "arrays":
                compressed_record,

            "metrics":
                metrics,

            "compression":
                compression_info,
        }
    )

    return Message(
        content=content,
        reply_to=msg,

    )
