"""Compatibility entry point for the historical prototype and dummy smoke.

Published YOLO26 training uses ``python -m runners.train_d3``.
"""
from legacy.prototype.runners.train import main


if __name__ == '__main__':
    main()
