from logging import getLogger
from typing import Any, Dict, List, Union

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool

from ..primitives.base import Primitive

_logger = getLogger(__name__)

_REINIT_IGNORED_KWARGS = {"params"}

SUBPARAM_SEP = "$$"


def _resolve_subparam_key(name: str, keep_mask) -> "str | None":
    """Resolve a ``keep_mask`` key for an optimizer group named ``name``.

    A group name matches a ``keep_mask`` key if they are equal or if the
    group name has the form ``key$$subparam_name`` (with ``$$`` as the
    separator). The separator is a non-Python-identifier character so it
    cannot collide with any batched-parameter name produced by
    ``Primitive.param_groups``.
    """
    if not isinstance(keep_mask, dict):
        return None
    if name in keep_mask:
        return name
    if SUBPARAM_SEP in name:
        prefix = name.split(SUBPARAM_SEP, 1)[0]
        if prefix in keep_mask:
            return prefix
    return None


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
        new_param_groups: List[Dict[str, Union[nn.Parameter, Any]]],
        keep_mask: Union[Bool[Tensor, "N"], Dict[str, Bool[Tensor, "N"]]],
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
            matched = False

            for g in new_param_groups:
                if g.get("name") == name:
                    _filter_group(
                        group,
                        g["params"],
                        current_mask,
                    )
                    matched = True
                    break  # IMPORTANT FIX

            if not matched:
                raise KeyError(f"No matching new_param_group for '{name}'")

        # --- state swap (safe version) ---
        self._optimizer.state = new_state

    # def filter(
    #     self,
    #     new_param_groups: List[Dict[str, Union[nn.Parameter, Any]]],
    #     keep_mask: Union[Bool[Tensor, "N"], Dict[str, Bool[Tensor, "N"]]],
    # ) -> None:
    #     """
    #     Filter optimizer state after parameter culling.

    #     Assumes:
    #     - positional correspondence between old and new params
    #     - first-dimension pruning only
    #     """

    #     new_state = {}

    #     def _filter_group(group, new_params, mask):
    #         old_params = group["params"]
    #         new_params_list = list(new_params)

    #         if len(old_params) != len(new_params_list):
    #             raise ValueError(
    #                 f"Parameter mismatch in group '{group.get('name')}': "
    #                 f"{len(old_params)} -> {len(new_params_list)}"
    #             )

    #         param_iter = iter(new_params_list)
    #         new_group_params = []

    #         for old_p in old_params:
    #             new_p = next(param_iter)
    #             new_group_params.append(new_p)

    #             # transfer optimizer state if it exists
    #             if old_p in self._optimizer.state:
    #                 old_state = self._optimizer.state[old_p]
    #                 new_state_for_p = {}

    #                 for k, v in old_state.items():
    #                     if (
    #                         isinstance(v, torch.Tensor)
    #                         and v.ndim > 0
    #                         and v.shape[0] == mask.shape[0]
    #                     ):
    #                         new_state_for_p[k] = v[mask]
    #                     else:
    #                         new_state_for_p[k] = v

    #                 new_state[new_p] = new_state_for_p

    #         group["params"] = new_group_params

    #     # --- main loop over optimizer groups ---
    #     for group in self._optimizer.param_groups:
    #         name = group.get("name")

    #         if name is None:
    #             # fallback: assume first new group
    #             _filter_group(
    #                 group,
    #                 new_param_groups[0]["params"],
    #                 keep_mask,
    #             )
    #         elif name not in keep_mask:
    #             for p in group["params"]:
    #                 if p in self._optimizer.state:
    #                     new_state[p] = self._optimizer.state[p]
    #             continue
    #         else:
    #             matched = False

    #             for g in new_param_groups:
    #                 if g.get("name") == name:
    #                     _filter_group(
    #                         group,
    #                         g["params"],
    #                         keep_mask[name],
    #                     )
    #                     matched = True
    #                     break  # IMPORTANT FIX

    #             if not matched:
    #                 raise KeyError(f"No matching new_param_group for '{name}'")

    #     # --- state swap (safe version) ---
    #     self._optimizer.state = new_state

    def split(
        self,
        new_param_groups: List[Dict[str, Union[nn.Parameter, Any]]],
        split_mask: Union[Bool[Tensor, "N"], Dict[str, Bool[Tensor, "N"]]],
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
            matched = False

            for g in new_param_groups:
                if g.get("name") == name:
                    _split_group(
                        group,
                        g["params"],
                        current_mask,
                    )
                    matched = True
                    break

            if not matched:
                raise KeyError(f"No matching new_param_group for '{name}'")

        # --- state swap ---
        self._optimizer.state = new_state
