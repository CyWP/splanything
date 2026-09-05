from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool

from ..primitives.base import Primitive

_logger = getLogger(__name__)

_REINIT_IGNORED_KWARGS = {"params"}

SUBPARAM_SEP = "$$"
CHILD_SEP = "^^"


def _resolve_subparam_key(name: str, keep_mask) -> "str | None":
    """Resolve a ``keep_mask`` key for an optimizer group named ``name``.

    A group name matches a ``keep_mask`` key if they are equal or if the
    group name has the form ``key$$subparam_name`` (with ``$$`` as the
    separator). The separator is a non-Python-identifier character so it
    cannot collide with any batched-parameter name produced by
    ``Primitive.param_groups``.

    If the group name is prefixed with ``^^`` (it was registered through
    ``MetaPrimitive.param_groups`` as a child-primitive parameter) the
    prefix is stripped before matching. ``^^``-prefixed mask keys are
    ignored: they should not appear in practice and would otherwise mask
    legitimate matches.
    """
    if not isinstance(keep_mask, dict):
        return None
    bare_name = name[len(CHILD_SEP) :] if name.startswith(CHILD_SEP) else name
    if bare_name in keep_mask and not bare_name.startswith(CHILD_SEP):
        return bare_name
    if SUBPARAM_SEP in bare_name:
        prefix = bare_name.split(SUBPARAM_SEP, 1)[0]
        if prefix in keep_mask and not prefix.startswith(CHILD_SEP):
            return prefix
    return None


def _resolve_new_group_for_existing(
    existing_name: str,
    new_param_groups: "list[dict]",
    existing_param_groups: "list[dict] | None" = None,
) -> "tuple[dict | None, str]":
    """Match an existing optimizer group to a ``new_param_groups`` entry.

    Returns ``(new_group, action)`` where ``action`` is one of:

    - ``"update"``: replace the existing group's params with ``new_group``'s
      params and transfer optimizer state.
    - ``"preserve"``: keep the existing group's params and state unchanged.

    Matching rules respect two separators:

    - ``^^`` (CHILD_SEP) marks child-primitive parameters that were
      registered via ``MetaPrimitive.param_groups`` while leaving the
      child in charge of its own refinement. When both the existing and
      the new param group carry ``^^``, the match is ``preserve``: the
      meta-level call must not touch the child's params (its refinement
      rules run independently on the child primitive directly).
    - ``$$`` (SUBPARAM_SEP) marks sub-parameters of a ``MultiPrimitive``
      container (e.g. ``radial$$centroids``). It is used as a fallback
      when an existing ``^^radial$$centroids`` group must be matched
      against a child's bare ``centroids`` param group (when the user
      fires a refinement rule directly on the child).

    The cross-primitive guard is only active when the optimizer already
    holds some ``^^``-prefixed groups: that means the top-level
    primitive was a ``MetaPrimitive`` with ``primitive_trainable=True``
    and could also carry bare names from its own parameters that might
    collide with bare names coming from a child called directly. In
    that case, bare-vs-bare exact matches are skipped to avoid
    accidentally rewriting a meta-owned parameter with a child-owned
    one. Without that ambiguity (top-level ``MultiPrimitive`` or simple
    ``Primitive``), bare-vs-bare exact matches are honored as normal.
    """
    has_caret = existing_name.startswith(CHILD_SEP)
    bare_existing = existing_name[len(CHILD_SEP) :] if has_caret else existing_name
    last_component = None
    if has_caret and SUBPARAM_SEP in bare_existing:
        last_component = bare_existing.rsplit(SUBPARAM_SEP, 1)[1]

    existing_has_caret_groups = bool(existing_param_groups) and any(
        isinstance(g.get("name"), str) and g["name"].startswith(CHILD_SEP)
        for g in existing_param_groups
    )
    cross_primitive_guard = (
        existing_has_caret_groups
        and not any(
            isinstance(g.get("name"), str) and g["name"].startswith(CHILD_SEP)
            for g in new_param_groups
        )
        and not has_caret
    )

    for g in new_param_groups:
        g_name = g.get("name")
        if not isinstance(g_name, str):
            continue
        g_has_caret = g_name.startswith(CHILD_SEP)

        if has_caret and g_has_caret and g_name == existing_name:
            return g, "preserve"

        if has_caret and not g_has_caret:
            if g_name == bare_existing:
                return g, "update"
            if last_component is not None and g_name == last_component:
                return g, "update"

        if not has_caret and not g_has_caret and g_name == existing_name:
            if cross_primitive_guard:
                continue
            return g, "update"

    return None, "preserve"


class OptimizerWrapper:
    """Wrapper for torch.optim.Optimizer that supports reinitialization.

    Stores the optimizer class and recreation kwargs, allowing reinit()
    to recreate the optimizer with new parameters.

    Attributes:
        reinit: Function to reinitialize optimizer with new parameters.

    Usage:
        opt = OptimizerWrapper(optim.Adam, model.parameters(), lr=0.001)
        # ... after split ...
        opt.reinit(model.parameters())
    """

    def __init__(
        self,
        primitive: Primitive,
        optimizer_class: type[optim.Optimizer],
        **kwargs: Any,
    ):
        self._optimizer_class = optimizer_class
        self._reinit_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k not in _REINIT_IGNORED_KWARGS
        }
        self.reinit(primitive.param_groups())

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    @property
    def lr(self) -> float:
        g = self._optimizer.param_groups[0]
        lr = g["lr"]
        lr_mod = g.get("lr_modifier", None)
        if lr_mod is None:
            return lr
        return lr / lr_mod

    def reinit(self, param_groups: List[Dict[str, nn.Parameter]]) -> None:
        """Reinitialize optimizer with new parameters.

        Args:
            params: New parameters to optimize.
        """
        if "lr" in self._reinit_kwargs:
            base_lr = self._reinit_kwargs["lr"]
            for g in param_groups:
                lr_mod = g.get("lr_modifier", None)
                if lr_mod is None:
                    continue
                g["lr"] = base_lr * lr_mod
        self._optimizer = self._optimizer_class(param_groups, **self._reinit_kwargs)

    def zero_nan_grads(self) -> None:
        """Replace all NaN gradients with zeros.

        Call this after loss.backward() but before optimizer.step()
        to handle NaN gradients that can cause training issues.
        """
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    nan_mask = torch.isnan(p.grad)
                    if nan_mask.any():
                        _logger.warning("Nan found in gradients")
                        p.grad[nan_mask] = 0.0

    def step(self) -> None:
        """Perform a single optimization step.

        Automatically zero any NaN gradients before stepping.
        """
        self.zero_nan_grads()
        self._optimizer.step()

    def filter(
        self,
        new_param_groups: List[Dict[str, nn.Parameter | Any]],
        keep_mask: Bool[Tensor, "N"] | Dict[str, Bool[Tensor, "N"]],
    ) -> None:
        """
        Filter optimizer state after parameter culling.

        Assumes:
        - positional correspondence between old and new params
        - first-dimension pruning only
        """

        new_state = {}

        def _filter_group(group, new_params, mask):
            old_params = group["params"]

            # Wrap raw tensor/Parameter from param_groups() to match
            # PyTorch's internal list representation
            if isinstance(new_params, torch.Tensor):
                new_params_list = [new_params]
            else:
                new_params_list = list(new_params)

            if len(old_params) != len(new_params_list):
                raise ValueError(
                    f"Parameter mismatch in group '{group.get('name')}': "
                    f"{len(old_params)} -> {len(new_params_list)}"
                )

            param_iter = iter(new_params_list)
            new_group_params = []

            for old_p in old_params:
                new_p = next(param_iter)
                new_group_params.append(new_p)

                # transfer optimizer state if it exists
                if old_p in self._optimizer.state:
                    old_state = self._optimizer.state[old_p]
                    new_state_for_p = {}

                    for k, v in old_state.items():
                        if (
                            isinstance(v, torch.Tensor)
                            and v.ndim > 0
                            and v.shape[0] == mask.shape[0]
                        ):
                            new_state_for_p[k] = v[mask]
                        else:
                            new_state_for_p[k] = v

                    new_state[new_p] = new_state_for_p

            group["params"] = new_group_params

        # --- main loop over optimizer groups ---
        for group in self._optimizer.param_groups:
            name = group.get("name")

            if name is None:
                # fallback: assume first new group
                _filter_group(
                    group,
                    new_param_groups[0]["params"],
                    keep_mask,
                )
                continue

            matched_key = _resolve_subparam_key(name, keep_mask)
            if isinstance(keep_mask, dict) and matched_key is None:
                # No mask covers this group: preserve existing state.
                for p in group["params"]:
                    if p in self._optimizer.state:
                        new_state[p] = self._optimizer.state[p]
                continue

            current_mask = (
                keep_mask[matched_key]
                if isinstance(keep_mask, dict) and matched_key is not None
                else keep_mask
            )

            new_group, action = _resolve_new_group_for_existing(
                name,
                new_param_groups,
                existing_param_groups=self._optimizer.param_groups,
            )
            if action == "preserve":
                for p in group["params"]:
                    if p in self._optimizer.state:
                        new_state[p] = self._optimizer.state[p]
                continue

            if new_group is None:
                raise KeyError(f"No matching new_param_group for '{name}'")

            _filter_group(
                group,
                new_group["params"],
                current_mask,
            )

        # --- state swap (safe version) ---
        # Keep a defaultdict so PyTorch optimizers can lazily initialize
        # state (``state[p]`` -> ``{}``) for params not yet stepped.
        self._optimizer.state = defaultdict(dict, new_state)

    def split(
        self,
        new_param_groups: List[Dict[str, nn.Parameter | Any]],
        split_mask: Bool[Tensor, "N"] | Dict[str, Bool[Tensor, "N"]],
    ) -> None:
        """Expand optimizer state after parameter splitting/cloning.

        Assumes:
        - The original parameter array is preserved in place.
        - Newly generated splats are appended directly to the trailing end.
        """
        new_state = {}

        def _split_group(group, new_params, mask):
            old_params = group["params"]

            # FIX: If new_params is a raw tensor/Parameter from param_groups(),
            # wrap it in a list to match PyTorch's internal list structure.
            if isinstance(new_params, torch.Tensor):
                new_params_list = [new_params]
            else:
                new_params_list = list(new_params)

            if len(old_params) != len(new_params_list):
                raise ValueError(
                    f"Parameter mismatch in group '{group.get('name')}': "
                    f"{len(old_params)} -> {len(new_params_list)}"
                )

            param_iter = iter(new_params_list)
            new_group_params = []

            for old_p in old_params:
                new_p = next(param_iter)
                new_group_params.append(new_p)

                # Transfer and duplicate optimizer state if it exists
                if old_p in self._optimizer.state:
                    old_state = self._optimizer.state[old_p]
                    new_state_for_p = {}

                    for k, v in old_state.items():
                        if (
                            isinstance(v, torch.Tensor)
                            and v.ndim > 0
                            and v.shape[0] == mask.shape[0]
                        ):
                            # Keep the entire original state tensor intact
                            # Append the states of the newly generated splats at the end
                            new_elements_state = v[mask]

                            new_state_for_p[k] = torch.cat(
                                [v, new_elements_state], dim=0
                            )
                        else:
                            new_state_for_p[k] = v

                    new_state[new_p] = new_state_for_p

            group["params"] = new_group_params

        # --- main loop over optimizer groups ---
        for group in self._optimizer.param_groups:
            name = group.get("name")

            if name is None:
                _split_group(
                    group,
                    new_param_groups[0]["params"],
                    split_mask,
                )
                continue

            matched_key = _resolve_subparam_key(name, split_mask)
            if isinstance(split_mask, dict) and matched_key is None:
                # No mask covers this group: preserve existing state.
                for p in group["params"]:
                    if p in self._optimizer.state:
                        new_state[p] = self._optimizer.state[p]
                continue

            current_mask = (
                split_mask[matched_key]
                if isinstance(split_mask, dict) and matched_key is not None
                else split_mask
            )

            new_group, action = _resolve_new_group_for_existing(
                name,
                new_param_groups,
                existing_param_groups=self._optimizer.param_groups,
            )
            if action == "preserve":
                for p in group["params"]:
                    if p in self._optimizer.state:
                        new_state[p] = self._optimizer.state[p]
                continue

            if new_group is None:
                raise KeyError(f"No matching new_param_group for '{name}'")

            _split_group(
                group,
                new_group["params"],
                current_mask,
            )

        # --- state swap ---
        # Keep a defaultdict so PyTorch optimizers can lazily initialize
        # state (``state[p]`` -> ``{}``) for params not yet stepped.
        self._optimizer.state = defaultdict(dict, new_state)
