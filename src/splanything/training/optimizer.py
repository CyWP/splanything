from logging import getLogger
from typing import Any, Dict

import torch
import torch.optim as optim
from jaxtyping import Bool
from torch import Tensor

_logger = getLogger(__name__)

_REINIT_IGNORED_KWARGS = {"params"}


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
        optimizer_class: type[optim.Optimizer],
        params: Any,
        **kwargs: Any,
    ):
        self._optimizer_class = optimizer_class
        self._reinit_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k not in _REINIT_IGNORED_KWARGS
        }
        self._optimizer = optimizer_class(params, **kwargs)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    def reinit(self, params: Any) -> None:
        """Reinitialize optimizer with new parameters.

        Args:
            params: New parameters to optimize.
        """
        self._optimizer = self._optimizer_class(params, **self._reinit_kwargs)

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

    def filter(self, new_params_iter, keep_mask: Bool[Tensor, "N_pre_filter"]) -> None:
        """Filter optimizer state to match new parameters after culling.

        Receives the new parameters from the model and updates optimizer
        param_groups and state accordingly. State tensors are filtered
        to match the new parameter shapes.

        Args:
            new_params_iter: New parameters after culling (from p.parameters()).
        """
        new_params_list = list(new_params_iter)
        new_state = {}
        param_iter = iter(new_params_list)

        for group in self._optimizer.param_groups:
            new_group_params = []
            for p in group["params"]:
                if len(p.shape) > 0 and p.shape[0] > 1:
                    new_p = next(param_iter)
                    new_group_params.append(new_p)
                    if p in self._optimizer.state:
                        old_state = self._optimizer.state[p]
                        new_state_for_p = {}
                        for k, v in old_state.items():
                            if (
                                isinstance(v, torch.Tensor)
                                and len(v.shape) > 0
                                and v.shape[0] == keep_mask.shape[0]
                            ):
                                new_state_for_p[k] = v[keep_mask]
                            else:
                                new_state_for_p[k] = v
                        new_state[new_p] = new_state_for_p
                else:
                    new_p = next(param_iter)
                    new_group_params.append(new_p)
                    if p in self._optimizer.state:
                        new_state[p] = self._optimizer.state[p]

            group["params"] = new_group_params

        self._optimizer.state.clear()
        self._optimizer.state.update(new_state)

    # def split(self, new_params_iter, split_mask: Bool[Tensor, "N"]) -> None:
    #     """Update optimizer params and state for split primitives.

    #     Args:
    #         new_params_iter: New parameters after split.
    #         split_mask: Boolean tensor where True=was split (duplicated).
    #     """
    #     new_params_list = list(new_params_iter)
    #     split_indices = split_mask.nonzero().squeeze(-1)
    #     old_len = len(split_mask)
    #     split_count = len(split_indices)
    #     new_len = old_len + split_count

    #     new_state = {}
    #     params_iter = iter(new_params_list)
    #     for group in self._optimizer.param_groups:
    #         new_group_params = []
    #         batched_state = None
    #         expanded_batched_state = None

    #         for p in group["params"]:
    #             if len(p.shape) and p.shape[0] == old_len:
    #                 new_p = next(params_iter)
    #                 new_group_params.append(new_p)
    #                 if batched_state is None:
    #                     batched_state = self._optimizer.state.get(p, {})
    #                     expanded_batched_state = {}
    #                     for k, v in batched_state.items():
    #                         if (
    #                             isinstance(v, torch.Tensor)
    #                             and len(v.shape) > 0
    #                             and v.shape[0] == old_len
    #                         ):
    #                             expanded_batched_state[k] = torch.cat(
    #                                 [v, v[split_indices]], dim=0
    #                             )
    #                         else:
    #                             expanded_batched_state[k] = v
    #                     new_state[new_p] = expanded_batched_state
    #                 else:
    #                     new_state[new_p] = expanded_batched_state
    #             else:
    #                 new_p = next(params_iter)
    #                 new_group_params.append(new_p)
    #                 new_state[new_p] = self._optimizer.state.get(p, {})

    #         group["params"] = new_group_params

    #     self._optimizer.state.clear()
    #     self._optimizer.state.update(new_state)
