from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class Config:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "geometry": 0.15,
        "sequence": 0.15,
        "graph": 0.10,
        "dimensions": 0.15,
        "text": 0.15,
        "blocks": 0.10,
        "styles": 0.05,
        "decimal_precision": 0.05,
        "pieces": 0.10,
    })
    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")
    angle_bins: int = 36
    length_bins: int = 20
    spatial_grid: int = 10
    dimension_distance_bins: int = 10
    lcs_weight: float = 0.5
    levenshtein_weight: float = 0.5
    fingerprint_threshold: float = 0.8
    enable_prefilter: bool = True
    graph_parallel_tolerance_degrees: float = 2.0
    graph_perpendicular_tolerance_degrees: float = 2.0
    graph_tangency_tolerance: float = 1e-3
    graph_intersection_tolerance: float = 1e-6
    piece_adjacency_tolerance: float = 1e-3
    oda_converter_path: str = "ODAFileConverter"
    dwg_output_version: str = "ACAD2018"
    dwg_output_format: str = "DXF"
    dwg_temp_dir: Optional[Path] = None
