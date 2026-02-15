from typing import Any
from jaxtyping import Float

from torch import Tensor

def run_tensor_multiply(arr1: Float[list, "b x y"], arr2: Float[list, "b y z"]) -> Float[list, "b x z"]:
    raise NotImplementedError

def run_tensor_dot(arr1: Float[list, "..."], arr2: Float[list, "..."], dim: int):
    raise NotImplementedError

def get_sgd_cls() -> Any:
    raise NotImplementedError

def get_adam_cls() -> Any:
    raise NotImplementedError

def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    raise NotImplementedError

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    raise NotImplementedError