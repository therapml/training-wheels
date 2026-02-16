import numpy as np
import torch
import pytest

from .adapters import run_layernorm, run_rmsnorm
from .common import FIXTURES_PATH


class TestLayerNorm:
    @pytest.fixture
    def layernorm_data(self):
        input_data = np.load(FIXTURES_PATH / "layernorm_input.npy")
        gamma = np.load(FIXTURES_PATH / "layernorm_gamma.npy")
        beta = np.load(FIXTURES_PATH / "layernorm_beta.npy")
        return {
            "input": input_data,
            "gamma": gamma,
            "beta": beta,
        }

    def test_layernorm_output(self, layernorm_data):
        input_tensor = torch.from_numpy(layernorm_data["input"])
        gamma = torch.from_numpy(layernorm_data["gamma"])
        beta = torch.from_numpy(layernorm_data["beta"])

        result = run_layernorm(input_tensor, gamma, beta)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result
        
        # Check shape - output should match input shape
        assert result_np.shape == layernorm_data["input"].shape, \
            f"Expected shape {layernorm_data['input'].shape}, got {result_np.shape}"
        
        # Check correctness: LayerNorm = (x - mean) / sqrt(var + eps) * gamma + beta
        # Using PyTorch's LayerNorm for reference (normalized over last dimension)
        ln = torch.nn.LayerNorm(layernorm_data["input"].shape)
        ln.load_state_dict({
            "weight": gamma,
            "bias": beta,
        })
        expected = ln(input_tensor).detach().numpy()
        
        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg="LayerNorm output doesn't match expected computation"
        )


class TestRMSNorm:
    @pytest.fixture
    def rmsnorm_data(self):
        input_data = np.load(FIXTURES_PATH / "rmsnorm_input.npy")
        gamma = np.load(FIXTURES_PATH / "rmsnorm_gamma.npy")
        return {
            "input": input_data,
            "gamma": gamma,
        }

    def test_rmsnorm_output(self, rmsnorm_data):
        input_tensor = torch.from_numpy(rmsnorm_data["input"])
        gamma = torch.from_numpy(rmsnorm_data["gamma"])

        result = run_rmsnorm(input_tensor, gamma)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result
        
        # Check shape - output should match input shape
        assert result_np.shape == rmsnorm_data["input"].shape, \
            f"Expected shape {rmsnorm_data['input'].shape}, got {result_np.shape}"
        
        # Check correctness: RMSNorm = x / RMS(x) * gamma where RMS = sqrt(mean(x^2) + eps)
        # Using PyTorch's RMSNorm for reference
        rms = torch.nn.RMSNorm(rmsnorm_data["input"].shape)
        rms.load_state_dict({
            "weight": gamma,
        })
        expected = rms(input_tensor).detach().numpy()
        
        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg="RMSNorm output doesn't match expected computation"
        )
