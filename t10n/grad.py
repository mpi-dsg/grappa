#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import torch as th

from typing import cast, Tuple


def weighted_grad_hook(
        state: object,
        bucket: th.distributed.GradBucket) -> th.futures.Future[th.Tensor]:
    group_to_use = th.distributed.group.WORLD
    world_size = group_to_use.size()

    buffer = (cast(Tuple[th.Tensor, ...], bucket)[0] if isinstance(
        bucket, tuple) else bucket.buffer())

    #weighted_grad  = buffer.to(th.float16).div_(state['coarse_count'] * world_size)
    #weighted_grad  = buffer.to(th.float16).div_(state['coarse_count'] * world_size)
    
    # weighted_grad = buffer.div_( (float(state['coarse_count']) + 1.0 ) * float(world_size) )
    weighted_grad = buffer.mul_( float(state['coarse_count']) ).div_(float(world_size))
    print( "new_weight size: ", float(state['coarse_count']) )

    fut = th.distributed.all_reduce(weighted_grad, async_op=True).get_future()

    return fut.then(lambda fut: fut.value()[0])
