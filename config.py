from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Config:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "geometry": 0.35,
        "sequence": 0.25,
        "graph": 0.20,
        "styles": 0.10,
        "metadata": 0.10,
    })
    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")
    angle_bins: int = 36
    length_bins: int = 20
    spatial_grid: int = 10
    lcs_weight: float = 0.5
    levenshtein_weight: float = 0.5
    fingerprint_threshold: float = 0.8
    enable_prefilter: bool = True
    graph_parallel_tolerance_degrees: float = 2.0
    graph_perpendicular_tolerance_degrees: float = 2.0
    graph_tangency_tolerance: float = 1e-3
    graph_intersection_tolerance: float = 1e-6
