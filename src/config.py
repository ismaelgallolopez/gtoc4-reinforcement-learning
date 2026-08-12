from dataclasses import dataclass

@dataclass
class Config:
    asteroids_filepath: str = "data/gtoc4_problem_data.txt"
    n_asteroids: int | None = None  # None = load all
    control_interval: float = 86400.0  # s, time between control steps
    integration_substeps: int = 10  # RK4 sub-steps per control step
    seed: int = 0
