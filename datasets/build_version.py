"""Compatibility entry point for the historical illustrative v0/v1 builder.

Frozen D3 datasets use ``python -m datasets.build_d3``.
"""
from legacy.prototype.datasets.build_version import main


if __name__ == '__main__':
    main()
