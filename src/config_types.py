# gelecekte diğer configleri: PSOConfig, DBNConfig, PreprocessingConfig vs. gelebilir
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    """Training params"""
    n_samples: int = 800
    test_size: float = 0.2
    pso_retries: int = 3
    random_state: int = 42
    min_samples_per_image: int = 50
    
