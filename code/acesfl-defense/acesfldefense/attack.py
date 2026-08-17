"""Controlled Byzantine attacks for ACES-FL security experiments."""

from __future__ import annotations

from typing import Iterable


import numpy as np


SUPPORTED_ATTACKS = {
    "none",
    "sign_flip",
    "model_scale",
}


def poison_delta(
    delta_arrays: Iterable[np.ndarray],
    attack_type: str = "none",
    attack_scale: float = 5.0,
) -> list[np.ndarray]:
    """
    Apply a controlled model-poisoning attack to a client update.

    Parameters
    ----------
    delta_arrays:
        Local model deltas before communication compression.

    attack_type:
        none
            Return the legitimate update unchanged.

        sign_flip
            Reverse the update direction and amplify it:
                poisoned_delta = -attack_scale * delta

        model_scale

            Amplify the legitimate update without reversing it:
                poisoned_delta = attack_scale * delta

    attack_scale:
        Multiplicative attack strength.
    """

    attack_type = str(attack_type).lower().strip()

    if attack_type not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"Unsupported attack type: {attack_type}. "
            f"Supported: {sorted(SUPPORTED_ATTACKS)}"
        )

    arrays = [
        np.asarray(array, dtype=np.float32)
        for array in delta_arrays
    ]

    if attack_type == "none":
        return [
            array.copy()
            for array in arrays
        ]

    if attack_type == "sign_flip":
        return [
            (-float(attack_scale) * array)
            .astype(np.float32)
            for array in arrays
        ]


    if attack_type == "model_scale":
        return [
            (float(attack_scale) * array)
            .astype(np.float32)
            for array in arrays
        ]

    raise RuntimeError("Unreachable attack branch.")


def delta_l2_norm(
    delta_arrays: Iterable[np.ndarray],
) -> float:
    """Return global L2 norm of a model delta."""

    total = 0.0

    for array in delta_arrays:
        a = np.asarray(
            array,
            dtype=np.float64,
        )

        total += float(
            np.sum(a * a)
        )

    return float(
        np.sqrt(total)
    )
