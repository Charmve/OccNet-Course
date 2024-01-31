# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from .mmdet_train import custom_train_detector
from mmdet.apis import train_detector


def custom_train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    eval_model=None,
    meta=None,
):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    custom_train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        eval_model=eval_model,
        meta=meta,
    )


def train_model(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
    )
