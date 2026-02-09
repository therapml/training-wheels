from typing import List, Any
from jaxtyping import Float

def run_tensor_multiply(arr1: Float[List, "b x y"], arr2: Float[List, "b y z"]) -> Float[List, "b x z"]:
    raise NotImplementedError

def run_tensor_dot(arr1: Float[List, "..."], arr2: Float[List, "..."], dim: int):
    raise NotImplementedError

def get_sgd_cls() -> Any:
    raise NotImplementedError