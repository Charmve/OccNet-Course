"""
Author: Charmve yidazhang1@gmail.com
Date: 2024-01-28 22:56:00
LastEditors: Charmve yidazhang1@gmail.com
LastEditTime: 2024-01-28 23:03:57
FilePath: /OccNet-Course/Chapter08-FinalProject/OccNet/projects/mmdet3d_plugin/models/utils/bricks.py # noqa:E501
Version: 1.0.1
Blogs: charmve.blog.csdn.net
GitHub: https://github.com/Charmve
Description:

Copyright (c) 2023 by Charmve, All Rights Reserved.
Licensed under the MIT License.
"""

import time
from collections import defaultdict

import torch

time_maps = defaultdict(lambda: 0.0)
count_maps = defaultdict(lambda: 0.0)


def run_time(name):
    def middle(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            res = fn(*args, **kwargs)
            torch.cuda.synchronize()
            time_maps["%s : %s" % (name, fn.__name__)] += time.time() - start
            count_maps["%s : %s" % (name, fn.__name__)] += 1
            print(
                "%s : %s takes up %f "
                % (
                    name,
                    fn.__name__,
                    time_maps["%s : %s" % (name, fn.__name__)]
                    / count_maps["%s : %s" % (name, fn.__name__)],
                )
            )
            return res

        return wrapper

    return middle
