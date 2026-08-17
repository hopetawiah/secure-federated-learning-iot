import numpy as np

from flwr.app import ArrayRecord


def choose_compression_mode(
    bandwidth_mbps: float,
    low_threshold: float = 10.0,
    high_threshold: float = 30.0,
) -> str:
    """Choose adaptive precision from current bandwidth."""

    if bandwidth_mbps >= high_threshold:
        return "fp32"

    if bandwidth_mbps >= low_threshold:
        return "fp16"

    return "int8"


def compress_update(
    ndarrays,
    mode: str,
):
    """
    Compress model-update arrays.

    Returns:
        compressed_arrays

        metadata
    """

    mode = mode.lower()

    arrays = [
        np.asarray(a, dtype=np.float32)
        for a in ndarrays
    ]

    if mode == "fp32":

        compressed = [
            a.astype(np.float32)
            for a in arrays
        ]

        metadata = {
            "mode": "fp32",
            "scales": [],
        }

        return compressed, metadata

    if mode == "fp16":

        compressed = [
            a.astype(np.float16)
            for a in arrays
        ]

        metadata = {

            "mode": "fp16",
            "scales": [],
        }

        return compressed, metadata

    if mode == "int8":

        compressed = []
        scales = []

        for array in arrays:

            max_abs = float(
                np.max(np.abs(array))
            )

            if max_abs == 0.0:
                scale = 1.0
            else:
                scale = max_abs / 127.0

            quantized = np.clip(
                np.rint(array / scale),
                -127,
                127,
            ).astype(np.int8)

            compressed.append(quantized)
            scales.append(float(scale))

        metadata = {

            "mode": "int8",
            "scales": scales,
        }

        return compressed, metadata

    raise ValueError(
        f"Unsupported compression mode: {mode}"
    )


def decompress_update(
    compressed_arrays,
    metadata,
):
    """Restore compressed updates to float32."""

    mode = metadata["mode"].lower()

    if mode == "fp32":
        return [
            np.asarray(a, dtype=np.float32)
            for a in compressed_arrays
        ]

    if mode == "fp16":
        return [
            np.asarray(a, dtype=np.float32)
            for a in compressed_arrays
        ]

    if mode == "int8":


        scales = metadata["scales"]

        if len(scales) != len(compressed_arrays):
            raise ValueError(
                "INT8 scale count does not match array count."
            )

        return [
            (
                np.asarray(a, dtype=np.float32)
                * float(scale)
            )
            for a, scale in zip(
                compressed_arrays,
                scales,
                strict=True,
            )
        ]

    raise ValueError(
        f"Unsupported compression mode: {mode}"
    )


def raw_tensor_bytes(ndarrays) -> int:
    """Raw tensor bytes without serialization metadata."""

    return int(
        sum(
            np.asarray(a).nbytes
            for a in ndarrays

        )
    )


def flower_record_bytes(ndarrays) -> int:
    """
    Approximate Flower ArrayRecord serialized payload size.

    count_bytes() includes array bytes plus a small amount
    of serialization metadata.
    """

    record = ArrayRecord(
        list(ndarrays)
    )

    return int(
        record.count_bytes()
    )


def relative_l2_error(
    original,
    reconstructed,
) -> float:
    """Relative L2 reconstruction error."""

    original_norm_sq = 0.0
    error_norm_sq = 0.0

    for a, b in zip(
        original,

        reconstructed,
        strict=True,
    ):
        a = np.asarray(
            a,
            dtype=np.float64,
        )

        b = np.asarray(
            b,
            dtype=np.float64,
        )

        original_norm_sq += float(
            np.sum(a * a)
        )

        diff = a - b

        error_norm_sq += float(
            np.sum(diff * diff)
        )

    original_norm = np.sqrt(
        original_norm_sq
    )

    error_norm = np.sqrt(
        error_norm_sq
    )

    if original_norm == 0:

        return 0.0

    return float(
        error_norm / original_norm
    )
