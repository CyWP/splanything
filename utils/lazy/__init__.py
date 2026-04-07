"""Lazy property evaluation with dependency tracking.

Exposes:
- lazy_tree: Decorator for caching properties with automatic invalidation
- LazyTree: Core lazy evaluation tree class
- Node: Node in the dependency graph
- clear_all_caches: Clear caches for all registered lazy_tree instances
"""

from .core import lazy_tree, LazyTree, Node, clear_all_caches