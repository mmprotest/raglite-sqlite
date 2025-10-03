from dataclasses import dataclass


@dataclass
class Candidate:
    chunk_id: int
    score: float
