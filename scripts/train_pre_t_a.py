"""PRE-T-A compatibility entrypoint.

The official current training implementation lives in scripts/train.py. This
module remains only so PRE-T-A audit tests and historical commands resolve to
the same Skorp-Beta-0.2 / 11-feature implementation; it contains no separate
legacy training pipeline.
"""
try:
    from scripts.train import *  # noqa: F401,F403
    from scripts.train import main
except ModuleNotFoundError:  # direct `python scripts/train_pre_t_a.py` execution
    from train import *  # type: ignore # noqa: F401,F403
    from train import main  # type: ignore


if __name__ == '__main__':
    main()
