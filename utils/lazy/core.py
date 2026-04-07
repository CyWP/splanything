import ast
import inspect
import weakref

from contextlib import contextmanager
from functools import wraps
from typing import Dict, Set, Type

_REGISTRY: "weakref.Set[object]" = weakref.WeakSet()


def clear_all_caches() -> None:
    """Clear caches for all registered lazy_tree instances.

    Iterates through all objects decorated with @lazy_tree and clears
    their cached property values.
    """
    for obj in list(_REGISTRY):
        obj.clear_cache()


class Node:
    """Node in the lazy evaluation dependency graph.

    Attributes:
        name: Property name.
        compute: Function to compute the property value.
        deps: Set of property names this node depends on.
        children: Set of nodes that depend on this node.
        value: Cached value (None if dirty).
        dirty: True if value needs recomputation.
        is_var: True if this is a variable (not a computed property).
    """

    def __init__(self, name: str, compute=None, is_var=False):
        self.name = name
        self.compute = compute
        self.deps: Set[str] = set()
        self.children: Set["Node"] = set()
        self.value = None
        self.dirty = True
        self.is_var = is_var

    def mark_dirty(self):
        """Mark this node and all dependents as dirty."""
        if not self.dirty:
            self.dirty = True
            self.value = None
            for child in self.children:
                child.mark_dirty()


class LazyTree:
    """Lazy evaluation tree with dependency tracking.

    Manages a graph of properties where computed properties automatically
    invalidate when their dependencies change.

    Notes:
        - Use `detect_cycles()` to verify no circular dependencies.
        - Use `clear_cache()` to invalidate all cached values.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def ensure_var(self, name: str) -> Node:
        """Get or create a variable node.

        Args:
            name: Variable name.

        Returns:
            Node instance.
        """
        if name not in self.nodes:
            self.nodes[name] = Node(name, is_var=True)
        return self.nodes[name]

    def add_node(self, node: Node):
        """Add a node to the tree.

        Args:
            node: Node to add.
        """
        self.nodes[node.name] = node

    def build_children(self):
        """Build child dependencies from deps."""
        for node in list(self.nodes.values()):
            for dep in node.deps:
                if dep not in self.nodes:
                    self.ensure_var(dep)
                self.nodes[dep].children.add(node)

    def detect_cycles(self):
        """Check for circular dependencies.

        Raises:
            RuntimeError: If cycle detected.
        """
        visited = set()
        stack = set()

        def visit(n: Node):
            if n.name in stack:
                raise RuntimeError(f"Cycle detected involving '{n.name}'")
            if n.name in visited:
                return
            visited.add(n.name)
            stack.add(n.name)
            for d in n.deps:
                visit(self.nodes[d])
            stack.remove(n.name)

        for n in self.nodes.values():
            visit(n)

    def clear_cache(self):
        """Invalidate all cached values."""
        for n in self.nodes.values():
            n.mark_dirty()


def lazy_tree(cls: Type):
    """Decorator for lazy property evaluation with dependency tracking.

    Wraps a class to cache @property results and automatically invalidate
    when dependent properties change.

    Notes:
        - All properties must only depend on self attributes.
        - Setting an attribute marks dependent properties as dirty.
        - Use `clear_cache()` to manually invalidate.
        - Use `cache_disabled()` context manager for one-off computation.

    Example:
        @lazy_tree
        class MyClass:
            @property
            def computed(self) -> int:
                return self.base * 2  # depends on 'base'
    """
    source = inspect.getsource(cls)
    parsed = ast.parse(source)
    class_def = parsed.body[0]

    tree_template = LazyTree()

    for node in class_def.body:
        if isinstance(node, ast.FunctionDef):
            fn_obj = getattr(cls, node.name, None)
            if not isinstance(fn_obj, property):
                continue

            if len(node.args.args) != 1:
                raise TypeError(f"{node.name} must take only self")

            deps = set()

            class DepVisitor(ast.NodeVisitor):
                def visit_Attribute(self, n):
                    if isinstance(n.value, ast.Name) and n.value.id == "self":
                        deps.add(n.attr)
                    self.generic_visit(n)

            DepVisitor().visit(node)

            n = Node(node.name, compute=fn_obj.fget)
            n.deps = deps
            tree_template.add_node(n)

    tree_template.build_children()
    tree_template.detect_cycles()

    orig_init = cls.__init__

    @wraps(orig_init)
    def new_init(self, *a, **kw):
        self.__lazy_tree__ = LazyTree()
        for name, node in tree_template.nodes.items():
            c = Node(name, node.compute, node.is_var)
            c.deps = set(node.deps)
            self.__lazy_tree__.add_node(c)
        self.__lazy_tree__.build_children()
        _REGISTRY.add(self)
        orig_init(self, *a, **kw)

    cls.__init__ = new_init

    orig_setattr = cls.__setattr__

    def __setattr__(self, name, value):
        orig_setattr(self, name, value)
        tree = getattr(self, "__lazy_tree__", None)
        if not tree:
            return
        node = tree.ensure_var(name)
        node.value = value
        node.dirty = False
        for child in node.children:
            child.mark_dirty()

    cls.__setattr__ = __setattr__

    for name, node in tree_template.nodes.items():
        if not node.compute:
            continue

        def make_prop(nm):
            def getter(self):
                node = self.__lazy_tree__.nodes[nm]
                if node.dirty:
                    for dep in node.deps:
                        getattr(self, dep)
                    node.value = node.compute(self)
                    node.dirty = False
                return node.value

            return property(getter)

        setattr(cls, name, make_prop(name))

    def clear_cache(self):
        self.__lazy_tree__.clear_cache()

    cls.clear_cache = clear_cache

    @contextmanager
    def cache_disabled(self):
        tree = self.__lazy_tree__
        old = {k: n.dirty for k, n in tree.nodes.items()}
        try:
            tree.clear_cache()
            yield self
        finally:
            for k, v in old.items():
                tree.nodes[k].dirty = v

    cls.cache_disabled = cache_disabled

    return cls
