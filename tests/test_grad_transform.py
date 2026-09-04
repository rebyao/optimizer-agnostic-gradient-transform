import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_transform import signed_square, step_with_transformed_gradients


def _make_model_and_batch(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    x = torch.randn(16, 4)
    y = torch.randn(16, 3)
    return model, x, y


def _clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def _backward_pass(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = nn.functional.mse_loss(out, y)
    loss.backward()
    return loss


class TestSignedSquareCorrectness:
    def test_known_values(self):
        g = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = torch.tensor([-4.0, -0.25, 0.0, 0.25, 4.0])
        result = signed_square(g)
        assert torch.allclose(result, expected)

    def test_not_same_as_plain_square(self):
        # g.square() would lose the sign; signed_square must preserve it.
        g = torch.tensor([-3.0, 3.0])
        assert not torch.allclose(signed_square(g), g.square())


class TestSGDReferenceEquivalence:
    def test_sgd_matches_manual_transform(self):
        model_a, x, y = _make_model_and_batch(seed=1)
        model_b = _clone_model(model_a)

        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1, momentum=0.9)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1, momentum=0.9)

        for _ in range(3):
            _backward_pass(model_a, x, y)
            step_with_transformed_gradients(opt_a, signed_square)

            _backward_pass(model_b, x, y)
            with torch.no_grad():
                for p in model_b.parameters():
                    if p.grad is not None:
                        p.grad = signed_square(p.grad)
            opt_b.step()

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb)


class TestAdamWReferenceEquivalence:
    def test_adamw_matches_manual_transform_params_and_state(self):
        model_a, x, y = _make_model_and_batch(seed=2)
        model_b = _clone_model(model_a)

        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-2)

        for _ in range(3):
            _backward_pass(model_a, x, y)
            step_with_transformed_gradients(opt_a, signed_square)

            _backward_pass(model_b, x, y)
            with torch.no_grad():
                for p in model_b.parameters():
                    if p.grad is not None:
                        p.grad = signed_square(p.grad)
            opt_b.step()

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb)

        params_a = list(model_a.parameters())
        params_b = list(model_b.parameters())
        for pa, pb in zip(params_a, params_b):
            state_a = opt_a.state[pa]
            state_b = opt_b.state[pb]
            assert torch.allclose(state_a["exp_avg"], state_b["exp_avg"])
            assert torch.allclose(state_a["exp_avg_sq"], state_b["exp_avg_sq"])


class TestGradientRestoration:
    def test_grad_restored_to_backward_values(self):
        model, x, y = _make_model_and_batch(seed=3)
        _backward_pass(model, x, y)

        original_grads = {
            p: p.grad.clone() for p in model.parameters() if p.grad is not None
        }

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        step_with_transformed_gradients(opt, signed_square)

        for p, original in original_grads.items():
            assert p.grad is not None
            assert torch.equal(p.grad, original)


class TestNoneGradient:
    def test_unused_parameter_stays_none(self):
        model, x, y = _make_model_and_batch(seed=4)
        unused = nn.Linear(5, 5)

        opt = torch.optim.SGD(
            list(model.parameters()) + list(unused.parameters()), lr=0.1
        )

        _backward_pass(model, x, y)
        for p in unused.parameters():
            assert p.grad is None

        # Should not raise, and should not fabricate a gradient.
        step_with_transformed_gradients(opt, signed_square)

        for p in unused.parameters():
            assert p.grad is None


class _RaisingOptimizer:
    """Minimal fake optimizer whose step() always raises."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]

    def step(self, *args, **kwargs):
        raise RuntimeError("boom")


class TestExceptionSafety:
    def test_grads_restored_even_if_step_raises(self):
        model, x, y = _make_model_and_batch(seed=5)
        _backward_pass(model, x, y)

        original_grads = {
            p: p.grad.clone() for p in model.parameters() if p.grad is not None
        }

        fake_opt = _RaisingOptimizer(model.parameters())

        with pytest.raises(RuntimeError, match="boom"):
            step_with_transformed_gradients(fake_opt, signed_square)

        for p, original in original_grads.items():
            assert p.grad is not None
            assert torch.equal(p.grad, original)


class TestMultiParamGroup:
    def test_multiple_param_groups_all_transformed(self):
        model, x, y = _make_model_and_batch(seed=6)
        layer0_params = list(model[0].parameters())
        layer2_params = list(model[2].parameters())

        opt = torch.optim.SGD(
            [
                {"params": layer0_params, "lr": 0.1},
                {"params": layer2_params, "lr": 0.01},
            ]
        )

        _backward_pass(model, x, y)
        original_grads = {p: p.grad.clone() for p in model.parameters()}

        step_with_transformed_gradients(opt, signed_square)

        # After restore, every param across both groups must match backward grad.
        for p, original in original_grads.items():
            assert torch.equal(p.grad, original)

        # Sanity check: both groups actually received updates (params changed).
        assert len(opt.param_groups) == 2
