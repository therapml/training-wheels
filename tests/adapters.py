from typing import Any
from jaxtyping import Float

def run_tensor_multiply(arr1: Float[list, "b x y"], arr2: Float[list, "b y z"]) -> Float[list, "b x z"]:
    raise NotImplementedError

def run_tensor_dot(arr1: Float[list, "..."], arr2: Float[list, "..."], dim: int):
    raise NotImplementedError

def get_sgd_cls() -> Any:
    raise NotImplementedError