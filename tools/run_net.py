# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
from timesformer.visualization.tensorboard_vis import TensorboardWriter as visualize

from tools.test_net import test
from tools.train_net import train
from tools.val_net import val


def get_func(cfg):
    train_func = train
    test_func = test
    val_func = val
    return train_func, val_func, test_func

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    train, val, test = get_func(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.VAL.ENABLE and cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=val)
    elif cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
