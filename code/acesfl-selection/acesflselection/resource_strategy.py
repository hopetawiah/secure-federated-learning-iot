import csv
import random
import time
from pathlib import Path

from typing import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


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
        results_dir: str = "results",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clients_per_round = clients_per_round

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

        selected = ranked[
            : self.clients_per_round
        ]

        selected_set = set(selected)


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
                    self.participation[node_id],
                ])

        selected_labels = [
            self.node_labels[n]
            for n in selected
        ]

        print(
            f"\nACES ROUND {server_round}: "

            f"selected {len(selected)}/{len(node_ids)} "
            f"clients -> {selected_labels}"
        )

        config["server-round"] = server_round
        config["selection-method"] = (
            "ACES-Resource-Fairness"
        )

        record = RecordDict({
            self.arrayrecord_key: arrays,
            self.configrecord_key: config,
        })

        return [
            Message(
                content=record,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
            )
            for node_id in selected
        ]
