"""Test optimizer filter/split operations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from utils.pytorch import OptimizerWrapper


def test_optimizer_split_state_expansion():
    """Test that split() properly expands optimizer state."""
    torch.manual_seed(42)
    
    # Create a simple parameter batch
    N = 40
    params = torch.nn.Parameter(torch.randn(N))
    
    # Create optimizer wrapper
    opt = OptimizerWrapper(optim.Adam, [params], lr=0.001)
    
    # Do a dummy step to populate optimizer state
    loss = params.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # Check Adam state has expected shapes
    state = opt.state[params]
    print(f"Initial state keys: {state.keys()}")
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
    
    initial_exp_avg_shape = state['exp_avg'].shape
    initial_exp_avg_sq_shape = state['exp_avg_sq'].shape
    print(f"\nInitial exp_avg shape: {initial_exp_avg_shape}")
    
    # Simulate split: 3 primitives split, creating N+3=43 total
    new_N = N + 3
    split_mask = torch.zeros(N, dtype=torch.bool)
    split_mask[[5, 10, 15]] = True
    
    # Create new params after split (simulating what primitive.split does)
    new_params = torch.nn.Parameter(torch.randn(new_N))
    
    # Call split on optimizer
    print(f"\nCalling split with N={N}, split_count=3, new_N={new_N}")
    opt.split([new_params], split_mask)
    
    # Check state shapes match new params
    new_state = opt.state[new_params]
    print(f"\nNew state keys: {new_state.keys()}")
    for k, v in new_state.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
    
    assert new_state['exp_avg'].shape[0] == new_N, \
        f"Expected exp_avg shape[0]={new_N}, got {new_state['exp_avg'].shape[0]}"
    assert new_state['exp_avg_sq'].shape[0] == new_N, \
        f"Expected exp_avg_sq shape[0]={new_N}, got {new_state['exp_avg_sq'].shape[0]}"
    
    # Verify split positions have correct values (should be duplicated from original)
    orig_idx = torch.tensor([5, 10, 15])
    orig_values = state['exp_avg'][orig_idx]
    new_values = new_state['exp_avg'][N:]  # Last 3 elements
    print(f"\nOriginal values at split indices: {orig_values}")
    print(f"New values (should match): {new_values}")
    
    if torch.allclose(orig_values, new_values):
        print("✓ Split state expansion is correct!")
    else:
        print("✗ Split state expansion failed!")
        print(f"  Difference: {(orig_values - new_values).abs().max()}")
    
    print("\nTest passed!")


def test_optimizer_filter_state():
    """Test that filter() properly filters optimizer state."""
    torch.manual_seed(42)
    
    N = 40
    params = torch.nn.Parameter(torch.randn(N))
    opt = OptimizerWrapper(optim.Adam, [params], lr=0.001)
    
    # Do a dummy step
    loss = params.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # Filter: keep only 35 elements (remove 5)
    mask = torch.ones(N, dtype=torch.bool)
    mask[:5] = False
    
    new_params = params[mask].clone().detach().requires_grad_(True)
    new_params = torch.nn.Parameter(new_params)
    
    opt.filter(mask)
    
    # After filter, state should be updated
    print(f"\nAfter filter:")
    print(f"  param_groups[0] has {len(opt.param_groups[0]['params'])} params")
    print(f"  state has {len(opt.state)} entries")
    
    print("Test passed!")


if __name__ == "__main__":
    test_optimizer_split_state_expansion()
    print("\n" + "="*50 + "\n")
    test_optimizer_filter_state()