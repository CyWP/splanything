import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from primitives.cubic_grad import CubicGrad

cg = CubicGrad(size=5)

# Step by step: test R first, then ref_axis, then axes
print("1. cg.R...")
try:
    r = cg.R
    print(f"   OK: {r.shape}")
except Exception as e:
    print(f"   FAIL: {e}")

print("2. cg.ref_axis...")
try:
    ref = cg.ref_axis
    print(f"   OK: {ref.shape}")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback
    traceback.print_exc()

print("3. cg.axes...")
try:
    ax1, ax2 = cg.axes
    print(f"   OK: {ax1.shape}, {ax2.shape}")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback
    traceback.print_exc()

# Check what the axes property's compute function actually does
print("\n4. Direct compute test:")
tree = object.__getattribute__(cg, '__lazy_tree__')
axes_node = tree.nodes['axes']
print(f"   axes deps: {axes_node.deps}")
print(f"   axes dirty: {axes_node.dirty}")

print("5. Calling axes original fget directly:")
try:
    result = CubicGrad.__dict__['axes'].fget.fget(cg)
    print(f"   OK: {type(result)}")
except Exception as e:
    print(f"   FAIL: {type(e).__name__}: {e}")

print("6. Calling axes compute from lazy_tree:")
try:
    result = axes_node.compute(cg)
    print(f"   OK: {type(result)}")
except Exception as e:
    print(f"   FAIL: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()