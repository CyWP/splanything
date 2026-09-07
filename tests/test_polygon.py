"""Tests for the PolygonPrimitive primitive's geometry and integration."""

from __future__ import annotations

import math

import torch

from splanything.primitives import PolygonPrimitive


def _unit_polygon(n_sides: int = 6, theta: float = 0.0) -> PolygonPrimitive:
    """Single polygon at (0.5, 0.5) with sigma = alpha = 1."""
    p = PolygonPrimitive(size=1, n_sides=n_sides)
    with torch.no_grad():
        p.thetas.copy_(torch.tensor([theta]))
        p.sigma.copy_(torch.tensor([1.0]))
        p.alphas.copy_(torch.tensor([1.0]))
        p.centroids.copy_(torch.tensor([[0.5, 0.5]]))
    return p


def test_axes_regularly_spaced():
    """Axes are the reference axis rotated in regular 2*pi/n_sides steps."""
    p = _unit_polygon(n_sides=5, theta=0.3)
    axes = p.axes  # (1, 5, 2)
    assert axes.shape == (1, 5, 2)
    angles = torch.atan2(axes[0, :, 1], axes[0, :, 0])
    expected = torch.arange(5) * (2 * torch.pi / 5)
    diff = ((angles - 0.3 - expected + torch.pi) % (2 * torch.pi)) - torch.pi
    assert torch.allclose(diff, torch.zeros(5), atol=1e-5)
    assert torch.allclose(axes.norm(dim=-1), torch.ones(1, 5), atol=1e-6)


def test_projection_onto_closest_axis_is_positive():
    """Weight distance uses the max dot product: always the positive-aligned projection."""
    p = _unit_polygon(n_sides=6)
    co = torch.rand(500, 2)
    deltas = co - p.centroids[None, 0, :]
    axes = p.axes[0]  # (6, 2)
    dots = deltas @ axes.T  # (500, 6)
    proj = dots.amax(dim=-1)
    assert (proj >= 0).all()
    weights = p.sample_weights(co)
    expected = torch.exp(-(proj**2) / (2 * p.sigma**2 + 1e-8))
    assert torch.allclose(weights[:, 0], expected, atol=1e-6)


def test_weight_at_inradius_along_axis():
    """At distance sigma along an axis, the projection equals sigma."""
    p = _unit_polygon(n_sides=6, theta=0.0)
    co = torch.tensor([[0.5 + 1.0, 0.5]])  # along the reference axis
    assert torch.allclose(p.sample_weights(co), torch.exp(torch.tensor([-0.5])))


def test_level_set_is_regular_polygon():
    """The exp(-0.5) level set: inradius sigma on axes, circumradius at bisectors."""
    p = _unit_polygon(n_sides=6, theta=0.0)
    n = 6
    target = math.exp(-0.5)
    # On-axis distance for the target weight
    r_axis = 1.0
    # Bisector distance: proj = r * cos(pi/n)
    r_bis = r_axis / math.cos(math.pi / n)
    co_axis = torch.tensor([[0.5 + r_axis, 0.5]])
    co_bis = torch.tensor(
        [[0.5 + r_bis * math.cos(math.pi / n), 0.5 + r_bis * math.sin(math.pi / n)]]
    )
    assert torch.allclose(p.sample_weights(co_axis), torch.tensor([target]), atol=1e-6)
    assert torch.allclose(p.sample_weights(co_bis), torch.tensor([target]), atol=1e-6)
    # Intermediate directions fall inside the level set (polygon edges, not a circle)
    co_mid = torch.tensor(
        [
            [
                0.5 + r_axis * math.cos(math.pi / (2 * n)),
                0.5 + r_axis * math.sin(math.pi / (2 * n)),
            ]
        ]
    )
    w_mid = p.sample_weights(co_mid)
    assert w_mid > target * 1.001


def test_theta_rotates_the_polygon():
    """Rotating thetas rotates the level set with it."""
    theta = math.pi / 6
    p = _unit_polygon(n_sides=4, theta=theta)
    d = 1.0 / math.cos(math.pi / 4)  # bisector distance for the exp(-0.5) set
    co = torch.tensor(
        [[0.5, 0.5 + d]]
    )  # straight up: bisector for theta=0, axis for theta=pi/2
    # For theta = pi/6 the nearest axis is at pi/2 offset pi/6 from straight up
    w = p.sample_weights(co)
    proj = d * math.cos(math.pi / 6)
    expected = math.exp(-(proj**2) / 2)
    assert torch.allclose(w, torch.tensor([expected]), atol=1e-6)


def test_n_sides_validation():
    """n_sides < 3 is rejected."""
    for bad in (0, 1, 2, -4):
        try:
            PolygonPrimitive(size=1, n_sides=bad)
            raise AssertionError(f"expected ValueError for n_sides={bad}")
        except ValueError:
            pass


def test_sample_shapes_and_gradient_flow():
    """forward() shapes match the contract; gradients reach every parameter."""
    p = PolygonPrimitive(size=3, n_sides=5)
    co = torch.rand(16, 2)
    out = p(co)
    assert out.rgb.shape == (16, 3, 3)
    assert out.weights.shape == (16, 3)
    (out.weights.sum() + out.rgb.sum()).backward()
    grads = dict(p.named_grads())
    assert set(grads) == {"centroids", "thetas", "sigma", "color", "alphas"}
    assert all(torch.isfinite(g).all() for g in grads.values())
    # color only contributes to rgb; weights-only gradient touches the rest
    assert grads["color"].abs().sum() > 0
    assert grads["sigma"].abs().sum() > 0
    assert grads["thetas"].abs().sum() > 0


def test_areas_and_scales():
    """areas use the cutoff-level inradius polygon area; scales are isotropic."""
    p = PolygonPrimitive(size=4, n_sides=6)
    s1, s2 = p.scales
    assert torch.equal(s1, s2)
    expected = 6 * (p.sigma.detach() * p._sigma_cutoff) ** 2 * math.tan(math.pi / 6)
    assert torch.allclose(p.areas, expected, atol=1e-5)


def test_patch_mask_covers_cutoff_circumradius():
    """Patches within the cutoff circumradius are valid, farther ones are not."""
    p = _unit_polygon(n_sides=6)
    circum = p._sigma_cutoff * 1.0 / math.cos(math.pi / 6)
    centers = torch.tensor([[0.5 + circum - 0.05, 0.5], [0.5 + circum + 0.05, 0.5]])
    pm = p.patch_mask(
        centers=centers,
        patch_sizes=torch.tensor([1, 1]),
        H=torch.tensor([64, 64]),
        W=torch.tensor([64, 64]),
    )
    assert pm.shape == (2, 1)
    assert pm[0, 0] and not pm[1, 0]


def test_split_and_filter_integration():
    """Default splitter halves sigma on split rows; check_filter runs clean."""
    p = PolygonPrimitive(size=4, n_sides=5)
    orig_sigma = p.sigma.detach().clone()
    p.split(torch.tensor([True, False, True, False]))
    assert len(p) == 6
    assert torch.allclose(p.sigma[0], orig_sigma[0] / math.sqrt(2), atol=1e-6)
    assert torch.allclose(p.sigma[1], orig_sigma[1], atol=1e-6)
    assert p.check_filter() is None
