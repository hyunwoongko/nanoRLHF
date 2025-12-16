from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

