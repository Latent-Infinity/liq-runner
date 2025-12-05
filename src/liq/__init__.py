"""LIQ runner package root."""

from pkgutil import extend_path

# Enable namespace-style packaging so sibling liq-* libs can co-exist on PYTHONPATH
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
