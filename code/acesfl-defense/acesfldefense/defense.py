"""ACES-FL robust trust/anomaly screening."""

from __future__ import annotations

import numpy as np


def flatten_update(update):
    """Flatten a list of parameter tensors into one vector."""
    return np.concatenate(
        [
            np.asarray(x, dtype=np.float64).ravel()
            for x in update
        ]
    )


def l2_norm(update):
    """Global L2 norm of a model update."""
    v = flatten_update(update)
    return float(np.linalg.norm(v))


def cosine_similarity(update, reference):
    """Cosine similarity between two model updates."""
    a = flatten_update(update)
    b = flatten_update(reference)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)

    )

    if denom <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


def coordinate_median(updates):
    """
    Robust round consensus using coordinate-wise median.

    With a minority of Byzantine clients, the median is
    substantially less sensitive to extreme updates than
    an ordinary arithmetic mean.
    """
    if not updates:
        raise ValueError("updates cannot be empty")

    num_tensors = len(updates[0])
    result = []

    for i in range(num_tensors):
        stacked = np.stack(
            [
                np.asarray(
                    update[i],
                    dtype=np.float32,
                )
                for update in updates

            ],
            axis=0,
        )

        result.append(
            np.median(
                stacked,
                axis=0,
            ).astype(np.float32)
        )

    return result


def robust_norm_zscores(norms):
    """
    Robust modified z-score using median absolute deviation.
    """
    x = np.asarray(
        norms,
        dtype=np.float64,
    )

    median = float(
        np.median(x)
    )

    mad = float(
        np.median(
            np.abs(x - median)
        )
    )


    if mad <= 1e-12:
        # Avoid division by zero when benign norms are
        # almost identical.
        scale = max(
            abs(median) * 0.05,
            1e-12,
        )

        scores = (
            np.abs(x - median)
            / scale
        )
    else:
        scores = (
            0.6745
            * np.abs(x - median)
            / mad
        )

    return (
        scores.astype(float),
        median,
        mad,
    )


def screen_updates(
    updates,
    client_ids,
    trust_scores=None,
    cosine_threshold=0.20,

    norm_z_threshold=3.50,
    trust_threshold=0.50,
):
    """
    Detect abnormal client model updates.

    IMPORTANT:
    Detection uses only update behaviour and historical
    trust. It does NOT use the experiment's malicious label.

    A client is rejected if:
      1. its direction is inconsistent with robust consensus,
      2. its norm is a strong robust outlier, or
      3. its historical trust has already fallen too low.
    """

    if len(updates) != len(client_ids):
        raise ValueError(
            "updates/client_ids length mismatch"
        )

    if not updates:
        raise ValueError(
            "No updates supplied"
        )

    trust_scores = (
        trust_scores
        if trust_scores is not None
        else {}
    )


    consensus = coordinate_median(
        updates
    )

    norms = [
        l2_norm(update)
        for update in updates
    ]

    norm_z, median_norm, mad_norm = (
        robust_norm_zscores(norms)
    )

    results = []

    for i, (
        client_id,
        update,
        norm,
    ) in enumerate(
        zip(
            client_ids,
            updates,
            norms,
        )
    ):
        cosine = cosine_similarity(
            update,
            consensus,
        )

        trust = float(

            trust_scores.get(
                client_id,
                1.0,
            )
        )

        direction_anomaly = (
            cosine
            < cosine_threshold
        )

        norm_anomaly = (
            float(norm_z[i])
            > norm_z_threshold
        )

        trust_anomaly = (
            trust
            < trust_threshold
        )

        # ---------------------------------------------
        # ACES Defense V2
        #
        # Historical trust must NOT permanently lock
        # out a client whose current update is normal.
        #
        # Strong current anomalies are rejected
        # immediately.
        #
        # Low trust only contributes to rejection when
        # the current update is also moderately

        # suspicious.
        # ---------------------------------------------

        borderline_direction = (
            cosine
            < 0.60
        )

        borderline_norm = (
            float(norm_z[i])
            > 2.50
        )

        trust_supported_anomaly = bool(
            trust_anomaly
            and (
                borderline_direction
                or borderline_norm
            )
        )

        rejected = bool(
            direction_anomaly
            or norm_anomaly
            or trust_supported_anomaly
        )

        results.append(
            {
                "client_id":
                    client_id,


                "norm":
                    float(norm),

                "norm_z":
                    float(norm_z[i]),

                "cosine":
                    float(cosine),

                "trust_before":
                    trust,

                "direction_anomaly":
                    int(direction_anomaly),

                "norm_anomaly":
                    int(norm_anomaly),

                "trust_anomaly":
                    int(trust_anomaly),

                "borderline_direction":
                    int(borderline_direction),

                "borderline_norm":
                    int(borderline_norm),

                "trust_supported_anomaly":
                    int(trust_supported_anomaly),

                "accepted":
                    int(not rejected),


                "decision":
                    (
                        "ACCEPT"
                        if not rejected
                        else "REJECT"
                    ),
            }
        )

    return {
        "consensus":
            consensus,

        "median_norm":
            median_norm,

        "mad_norm":
            mad_norm,

        "results":
            results,
    }


def update_trust(
    old_trust,
    accepted,
    alpha=0.80,
):
    """
    Exponentially update historical trust.


    accepted -> target trust = 1
    rejected -> target trust = 0
    """
    target = (
        1.0
        if accepted
        else 0.0
    )

    new_trust = (
        float(alpha)
        * float(old_trust)
        + (
            1.0
            - float(alpha)
        )
        * target
    )

    return float(
        np.clip(
            new_trust,
            0.0,
            1.0,
        )
    )
