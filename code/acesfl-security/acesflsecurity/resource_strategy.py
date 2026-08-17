import csv
import random
import time

import numpy as np
import torch
from pathlib import Path
from typing import Iterable

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from acesflsecurity.compression import (
    choose_compression_mode,

    decompress_update,
)


class ResourceAwareFedAvg(FedAvg):
    """
    ACES-FL resource-aware client-selection strategy.

    Clients are ranked each round using:
      - Battery availability
      - CPU availability
      - Network bandwidth
      - Connectivity quality
      - Participation fairness

    Trust/security scoring is intentionally excluded from this
    experiment and will be introduced in a later ablation.
    """

    def __init__(
        self,
        clients_per_round: int = 10,
        fairness_slots: int = 2,
        compression_low_threshold_mbps: float = 10.0,
        compression_high_threshold_mbps: float = 30.0,
        results_dir: str = "results",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clients_per_round = clients_per_round

        self.fairness_slots = fairness_slots

        self.compression_low_threshold_mbps = float(
            compression_low_threshold_mbps
        )

        self.compression_high_threshold_mbps = float(
            compression_high_threshold_mbps
        )

        if not (
            0.0
            < self.compression_low_threshold_mbps
            < self.compression_high_threshold_mbps
        ):
            raise ValueError(
                "Compression thresholds must satisfy "
                "0 < low < high."
            )

        if fairness_slots < 0 or fairness_slots > clients_per_round:
            raise ValueError(
                "fairness_slots must be between 0 and clients_per_round"
            )

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed

        self.profiles = {}
        self.participation = {}

        self.last_selected = {}
        self.node_labels = {}

        self.selection_path = (
            self.results_dir / "selection_history.csv"
        )

        self.profile_path = (
            self.results_dir / "client_resource_profiles.csv"
        )

        self.compression_path = (
            self.results_dir / "compression_history.csv"
        )

        # Global model that was sent at the beginning
        # of the current federated round.
        self._current_global_state = None

        # Round-level measured upload statistics.
        self.round_upload_bytes = {}
        self.round_flower_payload_bytes = {}

    def _wait_for_nodes(self, grid: Grid):
        """Wait until all required clients are available."""

        deadline = time.time() + 30

        while True:
            node_ids = sorted(list(grid.get_node_ids()))

            if len(node_ids) >= self.min_available_nodes:

                return node_ids

            if time.time() > deadline:
                raise RuntimeError(
                    f"Only {len(node_ids)} nodes available; "
                    f"expected {self.min_available_nodes}."
                )

            time.sleep(0.2)

    def _initialize_profiles(self, node_ids):
        """Create reproducible heterogeneous IoT resource profiles."""

        if self.profiles:
            return

        rng = random.Random(self.seed)

        for label, node_id in enumerate(node_ids):

            battery = rng.uniform(0.35, 1.00)

            cpu_capacity = rng.choice(
                [0.25, 0.50, 1.00]
            )

            bandwidth_mbps = rng.uniform(
                2.0, 50.0
            )

            bandwidth_norm = (
                bandwidth_mbps - 2.0

            ) / 48.0

            connectivity = rng.uniform(
                0.60, 1.00
            )

            self.node_labels[node_id] = label

            self.profiles[node_id] = {
                "battery": battery,
                "cpu": cpu_capacity,
                "bandwidth": bandwidth_norm,
                "bandwidth_mbps": bandwidth_mbps,
                "connectivity": connectivity,
            }

            self.participation[node_id] = 0
            self.last_selected[node_id] = 0

        with self.profile_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "resource_node_label",
                "node_id",
                "base_battery",
                "base_cpu_capacity",
                "base_bandwidth_mbps",

                "base_connectivity",
            ])

            for node_id in node_ids:

                p = self.profiles[node_id]

                writer.writerow([
                    self.node_labels[node_id],
                    node_id,
                    p["battery"],
                    p["cpu"],
                    p["bandwidth_mbps"],
                    p["connectivity"],
                ])

    @staticmethod
    def _clip(value, low=0.0, high=1.0):
        return max(low, min(high, value))

    def _calculate_state(
        self,
        node_id,
        server_round,
    ):
        """
        Generate deterministic round-varying resource state.

        This models changing IoT resource availability while keeping
        experiments reproducible.
        """


        p = self.profiles[node_id]
        label = self.node_labels[node_id]

        rng = random.Random(
            self.seed
            + server_round * 10000
            + label
        )

        count = self.participation[node_id]

        battery = self._clip(
            p["battery"]
            - 0.01 * count
            + rng.uniform(-0.03, 0.03),
            0.15,
            1.0,
        )

        cpu = self._clip(
            p["cpu"]
            + rng.uniform(-0.10, 0.10),
            0.10,
            1.0,
        )

        bandwidth = self._clip(
            p["bandwidth"]
            + rng.uniform(-0.12, 0.12),
            0.02,
            1.0,
        )


        connectivity = self._clip(
            p["connectivity"]
            + rng.uniform(-0.08, 0.08),
            0.10,
            1.0,
        )

        idle_rounds = (
            server_round
            - self.last_selected[node_id]
        )

        fairness = self._clip(
            idle_rounds / 4.0
        )

        # ACES Resource-Fairness Score
        score = (
            0.30 * battery
            + 0.20 * cpu
            + 0.25 * bandwidth
            + 0.15 * connectivity
            + 0.10 * fairness
        )

        return {
            "battery": battery,
            "cpu": cpu,
            "bandwidth": bandwidth,
            "bandwidth_mbps": (
                2.0 + bandwidth * 48.0

            ),
            "connectivity": connectivity,
            "fairness": fairness,
            "score": score,
        }

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:

        node_ids = self._wait_for_nodes(grid)

        self._initialize_profiles(node_ids)

        # Preserve the current global model.
        # aggregate_train() will add the weighted
        # decompressed client delta to this model.
        self._current_global_state = {
            key: value.detach().cpu().clone()
            for key, value
            in arrays.to_torch_state_dict().items()
        }

        states = {}

        for node_id in node_ids:
            states[node_id] = self._calculate_state(
                node_id,

                server_round,
            )

        ranked = sorted(
            node_ids,
            key=lambda n: states[n]["score"],
            reverse=True,
        )

        # ACES Selection v2:
        # Most slots exploit strong resource availability,
        # while reserved fairness slots prevent starvation.
        resource_slots = (
            self.clients_per_round
            - self.fairness_slots
        )

        resource_selected = ranked[
            :resource_slots
        ]

        resource_set = set(resource_selected)

        remaining = [
            node_id
            for node_id in node_ids
            if node_id not in resource_set
        ]

        # Lowest participation first.
        # Ties favour the node idle for longest.
        # Resource score breaks any remaining ties.

        fairness_ranked = sorted(
            remaining,
            key=lambda n: (
                self.participation[n],
                -(
                    server_round
                    - self.last_selected[n]
                ),
                -states[n]["score"],
            ),
        )

        fairness_selected = fairness_ranked[
            :self.fairness_slots
        ]

        selected = (
            resource_selected
            + fairness_selected
        )

        selected_set = set(selected)

        # -------------------------------------------------
        # Controlled adversarial participation
        # -------------------------------------------------
        #
        # Exactly N of the clients already selected by
        # ACES are designated malicious for this round.
        #
        # The attack RNG is seeded by attack_seed + round,
        # making the experiment deterministic/reproducible.

        # -------------------------------------------------

        malicious_clients_per_round = int(
            config.get(
                "malicious-clients-per-round",
                0,
            )
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
                5.0,
            )
        )

        attack_seed = int(
            config.get(
                "attack-seed",
                4242,
            )
        )

        malicious_clients_per_round = max(
            0,

            min(
                malicious_clients_per_round,
                len(selected),
            ),
        )

        attack_rng = np.random.default_rng(
            attack_seed
            + server_round
        )

        if malicious_clients_per_round > 0:
            malicious_indices = (
                attack_rng.choice(
                    len(selected),
                    size=malicious_clients_per_round,
                    replace=False,
                )
            )

            malicious_selected = {
                selected[int(index)]
                for index
                in malicious_indices
            }
        else:
            malicious_selected = set()

        malicious_labels = sorted(
            self.node_labels[node_id]
            for node_id
            in malicious_selected

        )

        print(
            f"ACES ATTACK ROUND {server_round}: "
            f"malicious="
            f"{len(malicious_selected)}/"
            f"{len(selected)} | "
            f"type={attack_type} | "
            f"scale={attack_scale} | "
            f"labels={malicious_labels}"
        )

        selection_type = {
            n: "not_selected"
            for n in node_ids
        }

        for n in resource_selected:
            selection_type[n] = "resource"

        for n in fairness_selected:
            selection_type[n] = "fairness"

        # Adaptive compression decision using the
        # selected node's CURRENT round bandwidth.
        compression_mode = {}

        for node_id in selected:
            compression_mode[node_id] = (
                choose_compression_mode(
                    states[node_id][
                        "bandwidth_mbps"

                    ],
                    low_threshold=(
                        self.compression_low_threshold_mbps
                    ),
                    high_threshold=(
                        self.compression_high_threshold_mbps
                    ),
                )
            )

        # Update participation state
        for node_id in selected:

            self.participation[node_id] += 1
            self.last_selected[node_id] = (
                server_round
            )

        # Record every client's score every round
        write_header = (
            not self.selection_path.exists()
        )

        with self.selection_path.open(
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            if write_header:
                writer.writerow([

                    "round",
                    "resource_node_label",
                    "node_id",
                    "battery",
                    "cpu",
                    "bandwidth_mbps",
                    "connectivity",
                    "fairness",
                    "score",
                    "selected",
                    "selection_type",
                    "compression_mode",
                    "participation_count",
                ])

            for node_id in ranked:

                s = states[node_id]

                writer.writerow([
                    server_round,
                    self.node_labels[node_id],
                    node_id,
                    s["battery"],
                    s["cpu"],
                    s["bandwidth_mbps"],
                    s["connectivity"],
                    s["fairness"],
                    s["score"],
                    int(
                        node_id
                        in selected_set

                    ),
                    selection_type[node_id],
                    (
                        compression_mode[node_id]
                        if node_id in selected_set
                        else "not_selected"
                    ),
                    self.participation[node_id],
                ])

        resource_labels = [
            self.node_labels[n]
            for n in resource_selected
        ]

        fairness_labels = [
            self.node_labels[n]
            for n in fairness_selected
        ]

        print(
            f"\nACES V2 ROUND {server_round}: "
            f"selected {len(selected)}/{len(node_ids)} | "
            f"resource={resource_labels} | "
            f"fairness={fairness_labels}"
        )

        mode_counts = {
            "fp32": 0,
            "fp16": 0,
            "int8": 0,
        }


        messages = []

        for node_id in selected:

            mode = compression_mode[node_id]

            mode_counts[mode] += 1

            # Each selected node receives its own
            # bandwidth-dependent compression config.
            client_config = ConfigRecord(
                dict(config)
            )

            client_config["server-round"] = (
                server_round
            )

            client_config["selection-method"] = (
                "ACES-Resource-Fairness-v2"
            )

            client_config["compression-mode"] = (
                mode
            )

            client_config["bandwidth-mbps"] = (
                float(
                    states[node_id][
                        "bandwidth_mbps"
                    ]

                )
            )

            client_config["is-malicious"] = (
                int(
                    node_id
                    in malicious_selected
                )
            )

            client_config["attack-type"] = (
                attack_type
                if node_id
                in malicious_selected
                else "none"
            )

            client_config["attack-scale"] = (
                float(attack_scale)
                if node_id
                in malicious_selected
                else 0.0
            )

            client_config["attack-seed"] = (
                int(attack_seed)
            )

            record = RecordDict(
                {
                    self.arrayrecord_key:
                        arrays,


                    self.configrecord_key:
                        client_config,
                }
            )

            messages.append(
                Message(
                    content=record,
                    message_type=MessageType.TRAIN,
                    dst_node_id=node_id,
                )
            )

        print(
            f"ACES COMPRESSION ROUND "
            f"{server_round}: "
            f"FP32={mode_counts['fp32']} | "
            f"FP16={mode_counts['fp16']} | "
            f"INT8={mode_counts['int8']}"
        )

        return messages


    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[
        ArrayRecord | None,
        MetricRecord | None,

    ]:
        """
        Decompress client model DELTAS, compute their
        sample-weighted mean, then apply that mean update
        to the global model.

        This preserves FedAvg semantics:

            new_global =
                old_global
                + weighted_mean(client_delta)
        """

        # We intentionally disable Flower's normal
        # ArrayRecord consistency validation because
        # replies can contain FP32, FP16, or INT8
        # ArrayRecords plus an extra ConfigRecord.
        valid_replies, _ = (
            self._check_and_log_replies(
                replies,
                is_train=True,
                validate=False,
            )
        )

        if not valid_replies:
            return None, None

        # -------------------------------------------------
        # Record security evidence centrally on server.
        # This avoids concurrent CSV writes by Ray clients.
        # -------------------------------------------------


        attack_path = (
            self.compression_path.parent
            / "attack_history.csv"
        )

        attack_write_header = (
            not attack_path.exists()
        )

        with attack_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as attack_file:

            attack_writer = csv.writer(
                attack_file
            )

            if attack_write_header:
                attack_writer.writerow(
                    [
                        "round",
                        "partition_id",
                        "is_malicious",
                        "attack_type",
                        "attack_scale",
                        "clean_delta_norm",
                        "transmitted_delta_norm",
                        "attack_amplification",
                        "compression_mode",

                        "bandwidth_mbps",
                        "raw_update_bytes",
                        "compressed_array_bytes",
                    ]
                )

            for reply in valid_replies:

                content = reply.content

                compression_info = (
                    content["compression"]
                )

                client_metrics = (
                    content[
                        "metrics"
                    ]
                )

                attack_writer.writerow(
                    [
                        server_round,

                        int(
                            compression_info[
                                "partition-id"
                            ]
                        ),

                        int(
                            compression_info.get(

                                "is-malicious",
                                0,
                            )
                        ),

                        str(
                            compression_info.get(
                                "attack-type",
                                "none",
                            )
                        ),

                        float(
                            compression_info.get(
                                "attack-scale",
                                0.0,
                            )
                        ),

                        float(
                            client_metrics.get(
                                "clean_delta_norm",
                                0.0,
                            )
                        ),

                        float(
                            client_metrics.get(
                                "transmitted_delta_norm",
                                0.0,
                            )
                        ),


                        float(
                            client_metrics.get(
                                "attack_amplification",
                                0.0,
                            )
                        ),

                        str(
                            compression_info[
                                "mode"
                            ]
                        ),

                        float(
                            compression_info[
                                "bandwidth-mbps"
                            ]
                        ),

                        int(
                            client_metrics[
                                "raw_update_bytes"
                            ]
                        ),

                        int(
                            client_metrics[
                                "compressed_array_bytes"
                            ]
                        ),
                    ]

                )

        if self._current_global_state is None:
            raise RuntimeError(
                "Current global model was not saved "
                "before aggregation."
            )

        state_keys = list(
            self._current_global_state.keys()
        )

        weighted_sum = None
        total_examples = 0

        round_raw_bytes = 0
        round_compressed_bytes = 0
        round_flower_bytes = 0

        mode_counts = {
            "fp32": 0,
            "fp16": 0,
            "int8": 0,
        }

        compression_rows = []

        for msg in valid_replies:

            content = msg.content

            metrics = content["metrics"]


            compression_info = (
                content["compression"]
            )

            mode = str(
                compression_info["mode"]
            ).lower()

            scales = [
                float(value)
                for value
                in compression_info.get(
                    "scales",
                    [],
                )
            ]

            compressed_arrays = (
                content[self.arrayrecord_key]
                .to_numpy_ndarrays()
            )

            delta = decompress_update(
                compressed_arrays,
                {
                    "mode": mode,
                    "scales": scales,
                },
            )

            if len(delta) != len(state_keys):

                raise RuntimeError(
                    "Client update tensor count "
                    "does not match global model."
                )

            num_examples = int(
                metrics[self.weighted_by_key]
            )

            if weighted_sum is None:
                weighted_sum = [
                    np.zeros_like(
                        array,
                        dtype=np.float64,
                    )
                    for array in delta
                ]

            for index, array in enumerate(delta):
                weighted_sum[index] += (
                    np.asarray(
                        array,
                        dtype=np.float64,
                    )
                    * num_examples
                )

            total_examples += num_examples

            raw_bytes = int(
                metrics["raw_update_bytes"]
            )


            compressed_bytes = int(
                metrics[
                    "compressed_array_bytes"
                ]
            )

            flower_bytes = int(
                metrics[
                    "flower_compression_payload_bytes"
                ]
            )

            round_raw_bytes += raw_bytes

            round_compressed_bytes += (
                compressed_bytes
            )

            round_flower_bytes += (
                flower_bytes
            )

            mode_counts[mode] += 1

            node_id = int(
                msg.metadata.src_node_id
            )

            compression_rows.append(
                [
                    server_round,

                    self.node_labels.get(
                        node_id,
                        -1,
                    ),
                    node_id,
                    int(
                        compression_info[
                            "partition-id"
                        ]
                    ),
                    float(
                        compression_info[
                            "bandwidth-mbps"
                        ]
                    ),
                    mode,
                    num_examples,
                    raw_bytes,
                    compressed_bytes,
                    int(
                        metrics[
                            "flower_array_bytes"
                        ]
                    ),
                    int(
                        metrics[
                            "compression_metadata_bytes"
                        ]
                    ),
                    flower_bytes,
                    float(
                        metrics[

                            "compression_error"
                        ]
                    ),
                ]
            )

        if (
            total_examples <= 0
            or weighted_sum is None
        ):
            return None, None

        mean_delta = [
            (
                array
                / total_examples
            ).astype(np.float32)
            for array in weighted_sum
        ]

        # ---------------------------------------------
        # Apply aggregated delta to global model
        # ---------------------------------------------

        new_state = {}

        for key, delta_array in zip(
            state_keys,
            mean_delta,
            strict=True,
        ):


            global_tensor = (
                self._current_global_state[key]
            )

            delta_tensor = torch.from_numpy(
                delta_array
            ).to(
                dtype=global_tensor.dtype
            )

            new_state[key] = (
                global_tensor
                + delta_tensor
            )

        aggregated_arrays = ArrayRecord(
            new_state
        )

        # ---------------------------------------------
        # Aggregate normal client training metrics
        # ---------------------------------------------

        reply_contents = [
            msg.content
            for msg in valid_replies
        ]

        aggregated_metrics = (
            self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,

            )
        )

        raw_reduction = (
            1.0
            - (
                round_compressed_bytes
                / round_raw_bytes
            )
        ) * 100.0

        self.round_upload_bytes[
            server_round
        ] = round_compressed_bytes

        self.round_flower_payload_bytes[
            server_round
        ] = round_flower_bytes

        aggregated_metrics[
            "round_raw_update_bytes"
        ] = int(round_raw_bytes)

        aggregated_metrics[
            "round_compressed_update_bytes"
        ] = int(round_compressed_bytes)

        aggregated_metrics[
            "round_flower_compression_bytes"
        ] = int(round_flower_bytes)

        aggregated_metrics[

            "compression_reduction"
        ] = float(raw_reduction / 100.0)

        aggregated_metrics[
            "fp32_clients"
        ] = int(mode_counts["fp32"])

        aggregated_metrics[
            "fp16_clients"
        ] = int(mode_counts["fp16"])

        aggregated_metrics[
            "int8_clients"
        ] = int(mode_counts["int8"])

        # ---------------------------------------------
        # Persist client-level compression evidence
        # ---------------------------------------------

        write_header = (
            not self.compression_path.exists()
        )

        with self.compression_path.open(
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            if write_header:
                writer.writerow(

                    [
                        "round",
                        "resource_node_label",
                        "node_id",
                        "partition_id",
                        "bandwidth_mbps",
                        "compression_mode",
                        "num_examples",
                        "raw_update_bytes",
                        "compressed_array_bytes",
                        "flower_array_bytes",
                        "compression_metadata_bytes",
                        "flower_compression_payload_bytes",
                        "compression_error",
                    ]
                )

            writer.writerows(
                compression_rows
            )

        print(
            f"\nCOMPRESSION ROUND {server_round}: "
            f"FP32={mode_counts['fp32']} | "
            f"FP16={mode_counts['fp16']} | "
            f"INT8={mode_counts['int8']} | "
            f"Raw={round_raw_bytes:,} B | "
            f"Compressed="
            f"{round_compressed_bytes:,} B | "
            f"Reduction={raw_reduction:.2f}%"
        )


        return (
            aggregated_arrays,
            aggregated_metrics,
        )
