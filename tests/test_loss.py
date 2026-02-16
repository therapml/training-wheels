import numpy as np
import torch
import pytest

from .adapters import run_cross_entropy_loss
from .common import FIXTURES_PATH


class TestCrossEntropyLoss:
    @pytest.fixture
    def ce_loss_data(self):
        y = np.load(FIXTURES_PATH / "cross_entropy_y.npy")
        y_bar = np.load(FIXTURES_PATH / "cross_entropy_y_bar.npy")
        return {
            "y": y,
            "y_bar": y_bar,
        }

    def test_cross_entropy_loss_output(self, ce_loss_data):
        y_tensor = torch.from_numpy(ce_loss_data["y"])
        y_bar_tensor = torch.from_numpy(ce_loss_data["y_bar"])
        
        result = run_cross_entropy_loss(y_tensor, y_bar_tensor)
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        
        # Result should be a scalar or 1D tensor with batch size
        # Cross entropy typically returns loss per sample or a single scalar
        assert result_np.ndim <= 1, \
            f"Expected scalar or 1D output, got shape {result_np.shape}"
        
        # Check correctness: cross entropy = -sum(y_bar * log(softmax(y)))
        # Manual computation
        y_softmax = torch.nn.functional.softmax(y_tensor, dim=1)
        log_softmax = torch.log(y_softmax)
        expected = -(y_bar_tensor * log_softmax).sum(dim=1).mean()
        
        np.testing.assert_allclose(
            result_np,
            expected.numpy(),
            rtol=1e-4,
            atol=1e-6,
            err_msg="Cross entropy loss doesn't match manual computation"
        )
