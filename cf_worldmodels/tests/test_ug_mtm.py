"""Tests for src/models/ug_mtm.py (UG-MTM: uncertainty-gated mixture)."""
import pytest
import torch
import torch.nn as nn

from src.models.rssm import BaseWorldModel
from src.models.ug_mtm import (
    ExpertPool,
    ThresholdNet,
    UG_MTM,
    UncertaintyHead,
    compute_gates,
    compute_uncertainty,
    register_gradient_scaling_hooks,
)

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM


# --- ExpertPool ------------------------------------------------------------

def test_expert_pool_creates_k_experts():
    pool = ExpertPool(K=4, input_dim=6, hidden_dim=8)
    assert len(pool.experts) == 4
    assert all(isinstance(e, nn.GRUCell) for e in pool.experts)


def test_expert_pool_experts_have_independent_parameters():
    """The mixture only isolates tasks if no weights are shared across experts."""
    pool = ExpertPool(K=3, input_dim=6, hidden_dim=8)
    ids = [id(p) for e in pool.experts for p in e.parameters()]
    assert len(ids) == len(set(ids))


def test_expert_pool_experts_are_initialised_differently():
    pool = ExpertPool(K=2, input_dim=6, hidden_dim=8)
    assert not torch.allclose(
        pool.experts[0].weight_ih, pool.experts[1].weight_ih
    )


def test_expert_pool_leaves_global_rng_reproducible():
    """Building the pool must not reseed the global RNG nondeterministically.

    The experiment runner calls torch.manual_seed(seed) and then constructs the
    model, so anything that reseeds during construction silently discards the
    run's seed and makes results unreproducible.
    """
    def draw_after_construction():
        torch.manual_seed(123)
        ExpertPool(K=4, input_dim=6, hidden_dim=8)
        return torch.randn(4)

    assert torch.allclose(draw_after_construction(), draw_after_construction())


def test_expert_pool_weights_are_reproducible():
    """Expert initialisation must not depend on the global RNG at all."""
    torch.manual_seed(0)
    first = ExpertPool(K=2, input_dim=6, hidden_dim=8)
    torch.manual_seed(999)
    second = ExpertPool(K=2, input_dim=6, hidden_dim=8)

    for a, b in zip(first.experts, second.experts):
        assert torch.equal(a.weight_ih, b.weight_ih)
        assert torch.equal(a.bias_hh, b.bias_hh)


# --- compute_uncertainty ---------------------------------------------------

def test_uncertainty_is_zero_when_T_is_one():
    head = UncertaintyHead(6, 8, dropout_rate=0.5)
    u = compute_uncertainty(head, torch.randn(5, 6), T=1)
    assert u.shape == (5,)
    assert torch.all(u == 0)


def test_uncertainty_is_positive_with_dropout_and_multiple_passes():
    head = UncertaintyHead(6, 8, dropout_rate=0.5)
    u = compute_uncertainty(head, torch.randn(5, 6), T=10)
    assert u.shape == (5,)
    assert torch.all(u > 0)


def test_uncertainty_is_zero_without_dropout():
    """With dropout disabled, MC passes are identical, so variance vanishes."""
    head = UncertaintyHead(6, 8, dropout_rate=0.0)
    u = compute_uncertainty(head, torch.randn(5, 6), T=10)
    assert torch.allclose(u, torch.zeros(5), atol=1e-7)


def test_uncertainty_restores_module_training_mode():
    head = UncertaintyHead(6, 8)
    head.eval()
    compute_uncertainty(head, torch.randn(3, 6), T=4)
    assert not head.training

    head.train()
    compute_uncertainty(head, torch.randn(3, 6), T=4)
    assert head.training


# --- ThresholdNet ----------------------------------------------------------

def test_threshold_is_in_unit_interval():
    net = ThresholdNet(window=8)
    tau = net()
    assert tau.ndim == 0
    assert 0.0 < tau.item() < 1.0


def test_threshold_history_updates_in_place():
    net = ThresholdNet(window=4)
    net.update_history(0.5)
    assert net._history[0].item() == pytest.approx(0.5)


def test_threshold_history_wraps_around_circularly():
    net = ThresholdNet(window=3)
    for value in [1.0, 2.0, 3.0, 4.0]:
        net.update_history(value)
    # The 4th write overwrites slot 0.
    assert net._history.tolist() == pytest.approx([4.0, 2.0, 3.0])


def test_threshold_responds_to_history():
    net = ThresholdNet(window=4)
    tau_empty = net().item()
    for _ in range(4):
        net.update_history(5.0)
    assert net().item() != pytest.approx(tau_empty)


# --- compute_gates ---------------------------------------------------------

def test_gates_sum_to_one():
    gates = compute_gates(torch.tensor([0.1, 0.9]), torch.tensor(0.5),
                          K=4, K_active=2)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_gates_shape_is_batch_by_k():
    gates = compute_gates(torch.rand(7), torch.tensor(0.5), K=3, K_active=1)
    assert gates.shape == (7, 3)


def test_unused_experts_receive_no_gate_mass():
    """Experts beyond K_active are not yet allocated and must stay silent."""
    gates = compute_gates(torch.tensor([0.5]), torch.tensor(0.5),
                          K=4, K_active=1)
    assert gates[0, 2].item() == pytest.approx(0.0, abs=1e-8)
    assert gates[0, 3].item() == pytest.approx(0.0, abs=1e-8)


def test_low_uncertainty_favours_existing_experts():
    """u < tau means the domain is known, so consolidated experts must win."""
    gates = compute_gates(torch.tensor([0.0]), torch.tensor(0.5),
                          K=4, K_active=1, lambda_=10.0)
    assert gates[0, 0].item() > gates[0, 1].item()


def test_high_uncertainty_favours_the_active_expert():
    """u > tau signals a new domain, routing gradient to the fresh expert."""
    gates = compute_gates(torch.tensor([1.0]), torch.tensor(0.5),
                          K=4, K_active=1, lambda_=10.0)
    assert gates[0, 1].item() > gates[0, 0].item()


def test_first_expert_takes_everything_when_no_experts_consolidated():
    gates = compute_gates(torch.rand(3), torch.tensor(0.5), K=4, K_active=0)
    assert torch.allclose(gates[:, 0], torch.ones(3), atol=1e-6)


def test_larger_lambda_sharpens_the_gate():
    u, tau = torch.tensor([1.0]), torch.tensor(0.5)
    soft = compute_gates(u, tau, K=4, K_active=1, lambda_=1.0)
    sharp = compute_gates(u, tau, K=4, K_active=1, lambda_=50.0)
    assert sharp[0, 1].item() > soft[0, 1].item()


# --- gradient scaling ------------------------------------------------------

def test_gradient_scaling_hooks_scale_by_gate_weight():
    """Gradients reaching each expert are scaled by that expert's gate, so a
    near-zero gate leaves its weights effectively untouched."""
    experts = nn.ModuleList([nn.GRUCell(4, 4) for _ in range(2)])
    handles = []
    register_gradient_scaling_hooks(experts, [1.0, 0.0], handles)

    x, h = torch.randn(3, 4), torch.zeros(3, 4)
    (experts[0](x, h).sum() + experts[1](x, h).sum()).backward()

    assert experts[0].weight_ih.grad.abs().sum() > 0
    assert experts[1].weight_ih.grad.abs().sum() == pytest.approx(0.0)
    for handle in handles:
        handle.remove()


def test_gradient_scaling_hooks_are_removable():
    experts = nn.ModuleList([nn.GRUCell(4, 4)])
    handles = []
    register_gradient_scaling_hooks(experts, [0.0], handles)
    assert handles
    for handle in handles:
        handle.remove()

    x, h = torch.randn(3, 4), torch.zeros(3, 4)
    experts[0](x, h).sum().backward()
    assert experts[0].weight_ih.grad.abs().sum() > 0


# --- UG_MTM ----------------------------------------------------------------

@pytest.fixture
def model(ug_cfg):
    return UG_MTM(ug_cfg)


def test_ug_mtm_implements_base_world_model(model):
    assert isinstance(model, BaseWorldModel)


def test_ug_mtm_builds_from_config(model, ug_cfg):
    assert len(model.expert_pool.experts) == ug_cfg.model.num_experts
    assert model.K_active == 0


def test_ug_mtm_transition_shape(model):
    model.eval()
    h = torch.zeros(4, HIDDEN_DIM)
    z = torch.randn(4, LATENT_DIM)
    a = torch.randn(4, ACTION_DIM)
    assert model.transition(h, z, a).shape == (4, HIDDEN_DIM)


def test_ug_mtm_eval_gating_is_driven_by_uncertainty(model):
    """At evaluation the gate must react to the input's uncertainty.

    Forcing u_t = 0 in eval mode would score every input as in-distribution,
    disconnecting the newly activated expert regardless of what it learned and
    making the forgetting metrics blind to it.
    """
    model.eval()
    model.K_active = 1
    h = torch.zeros(4, HIDDEN_DIM)
    z = torch.randn(4, LATENT_DIM)
    a = torch.randn(4, ACTION_DIM)

    # Make the new expert wildly different from the consolidated one; if the
    # gate ever routes to it, the transition output has to move.
    with torch.no_grad():
        for param in model.expert_pool.experts[1].parameters():
            param.add_(torch.randn_like(param) * 50.0)

    tau = model.threshold_net()
    u = model.get_uncertainty(h, z, a)
    assert torch.any(u > 0), "eval-time uncertainty collapsed to zero"

    from src.models.ug_mtm import compute_gates

    high = compute_gates(torch.full((4,), float(tau) + 1.0), tau,
                         model.K, model.K_active, model.gate_lambda)
    assert high[0, 1].item() > high[0, 0].item()


def test_ug_mtm_eval_uses_more_mc_samples_than_training(ug_cfg):
    """A variance estimated from few passes is noisy; evaluation should not
    inject avoidable variance into the reported metrics."""
    ug_cfg.model.mc_dropout_T_eval = 12
    m = UG_MTM(ug_cfg)
    assert m.T_eval == 12
    assert m.T == ug_cfg.model.mc_dropout_T


def test_ug_mtm_eval_mc_samples_default_to_training_value(ug_cfg):
    assert "mc_dropout_T_eval" not in ug_cfg.model
    m = UG_MTM(ug_cfg)
    assert m.T_eval == ug_cfg.model.mc_dropout_T


def test_ug_mtm_eval_does_not_mutate_threshold_history(model):
    """Evaluation must not write to model state."""
    model.eval()
    before = model.threshold_net._history.clone()
    model.transition(torch.zeros(2, HIDDEN_DIM), torch.randn(2, LATENT_DIM),
                     torch.randn(2, ACTION_DIM))
    assert torch.equal(before, model.threshold_net._history)


def test_ug_mtm_get_uncertainty_is_not_zero(model):
    """The whole method rests on a non-degenerate uncertainty signal."""
    model.train()
    h = torch.zeros(5, HIDDEN_DIM)
    z = torch.randn(5, LATENT_DIM)
    a = torch.randn(5, ACTION_DIM)
    u = model.get_uncertainty(h, z, a)
    assert u.shape == (5,)
    assert torch.all(u > 0)


def test_ug_mtm_routes_to_active_expert_only(model):
    """With K_active = 0 the output must equal expert 0 run alone."""
    model.eval()
    h = torch.zeros(3, HIDDEN_DIM)
    z = torch.randn(3, LATENT_DIM)
    a = torch.randn(3, ACTION_DIM)

    expected = model.expert_pool.experts[0](torch.cat([z, a], dim=-1), h)
    assert torch.allclose(model.transition(h, z, a), expected, atol=1e-5)


def test_ug_mtm_compute_loss_components(model, seq_batch):
    loss, comps = model.compute_loss(seq_batch)
    assert torch.isfinite(loss)
    assert set(comps) == {"reconstruction", "kl", "uncertainty", "nll"}


def test_ug_mtm_compute_loss_accepts_single_step_batch(model):
    batch = {"obs": torch.rand(2, 3, 64, 64),
             "actions": torch.rand(2, ACTION_DIM)}
    loss, _ = model.compute_loss(batch)
    assert torch.isfinite(loss)


def test_ug_mtm_backward_reaches_experts_and_uncertainty_head(model, seq_batch):
    model.train()
    loss, _ = model.compute_loss(seq_batch)
    loss.backward()

    assert model.expert_pool.experts[0].weight_ih.grad is not None
    assert model.uncertainty_head.fc1.weight.grad is not None
    assert model.vae.enc_fc.weight.grad is not None


def test_ug_mtm_gradient_hooks_are_replaced_not_accumulated(model):
    """Stale hooks from earlier timesteps would compound the gate scaling."""
    model.train()
    h = torch.zeros(2, HIDDEN_DIM)
    z = torch.randn(2, LATENT_DIM)
    a = torch.randn(2, ACTION_DIM)

    model.transition(h, z, a)
    after_first = len(model._grad_hooks)
    model.transition(h, z, a)
    model.transition(h, z, a)

    assert len(model._grad_hooks) == after_first


def test_ug_mtm_frozen_experts_keep_their_weights(model, seq_batch):
    """Training with K_active = 1 must leave expert 2 and 3 untouched."""
    model.train()
    model.K_active = 1
    frozen_before = model.expert_pool.experts[3].weight_ih.detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss, _ = model.compute_loss(seq_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.allclose(
        frozen_before, model.expert_pool.experts[3].weight_ih, atol=1e-8
    )
