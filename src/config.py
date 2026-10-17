from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MAPS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    48,
    50,
    51,
    53,
    54,
    56,
    57,
    58,
    59,
    60,
    61,
    65,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    76,
    78,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    88,
    90,
    92,
    94,
    95,
    97,
    98,
    99,
)

VAL_MAPS = (9, 55, 64, 75, 77, 79, 89, 91, 93, 100)

TEST_MAPS = (12, 24, 32, 47, 49, 52, 66, 87, 96)


@dataclass(frozen=True)
class DataConfig:
    prop_root: Path = PROJECT_ROOT / "assets" / "prop"
    building_root: Path = PROJECT_ROOT / "assets" / "buildings"
    los_root: Path = PROJECT_ROOT / "assets" / "los"
    cache_root: Path = PROJECT_ROOT / "assets" / "cache"
    rx_per_tx: int = 200
    spatial_bins: int = 10
    seed: int = 2026
    building_height: float = 20.0
    map_size: float = 256.0
    pad_vertex_id: int = 65535


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "full"
    maps: tuple[int, ...] = ()
    split_mode: str = "map"
    train_ratio: float = 0.8
    eval_split: str = "test"


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    fusion_dim: int = 32
    fusion_gate_groups: int = 8
    ray_layers: int = 2
    pointer_layers: int = 2
    candidate_layers: int = 3
    los_graph_dim: int = 128
    los_graph_layers: int = 3
    los_graph_heads: int = 4
    heads: int = 4
    dropout: float = 0.1
    k_pre: int = 64
    max_wall_tokens: int = 32
    feature_dim: int = 12
    endpoint_dim: int = 5
    ray_dim: int = 17
    wall_dim: int = 15
    ray_wall_dim: int = 34
    state_dim: int = 21


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 2e-4
    min_lr: float = 1e-6
    scheduler: str = "linear"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 2026
    decode_rx_limit: int = 4
    decode_interval: int = 1
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0


@dataclass(frozen=True)
class OverfitConfig:
    enabled: bool = False
    map_id: int = 0
    tx_count: int = 100
    rx_per_tx: int = 5


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    overfit: OverfitConfig = field(default_factory=OverfitConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[1]
    data_raw = raw.get("data", {})
    for key in ("prop_root", "building_root", "los_root", "cache_root"):
        if key in data_raw:
            value = Path(data_raw[key])
            data_raw[key] = value if value.is_absolute() else project_root / value
    experiment_raw = raw.get("experiment", {})
    if "maps" in experiment_raw:
        experiment_raw["maps"] = tuple(int(value) for value in experiment_raw["maps"])
    train_raw = raw.get("train", {})
    return AppConfig(
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**train_raw),
        experiment=ExperimentConfig(**experiment_raw),
        overfit=OverfitConfig(**raw.get("overfit", {})),
    )


def experiment_maps(cfg: AppConfig, split: str) -> tuple[int, ...]:
    if cfg.experiment.split_mode == "within_map":
        return cfg.experiment.maps
    return maps_for_split(split)


def split_for_map(map_id: int) -> str:
    if map_id in TRAIN_MAPS:
        return "train"
    if map_id in VAL_MAPS:
        return "val"
    if map_id in TEST_MAPS:
        return "test"
    raise ValueError(f"map_{map_id} is not part of the approved split")


def maps_for_split(split: str) -> tuple[int, ...]:
    return {"train": TRAIN_MAPS, "val": VAL_MAPS, "test": TEST_MAPS}[split]
