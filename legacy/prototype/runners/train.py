import argparse, yaml
from legacy.prototype.models.registry import get
from legacy.prototype.models.base import TrainRequest


def deep_merge(a: dict, b: dict):
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            deep_merge(a[k], v)
        else:
            a[k] = v

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--config", default="legacy/prototype/configs/base.yaml")
    p.add_argument("--override", default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.override:
        with open(args.override, "r", encoding="utf-8") as f:
            over = yaml.safe_load(f)
        deep_merge(cfg, over)

    model = get(args.model)
    model.train(TrainRequest(version=args.version, config=cfg))

if __name__ == "__main__":
    main()
