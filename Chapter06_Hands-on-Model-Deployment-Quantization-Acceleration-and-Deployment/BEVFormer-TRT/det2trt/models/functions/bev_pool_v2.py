from torch.autograd import Function
from third_party.bev_mmdet3d.ops.bev_pool_v2 import bev_pool_v2_gpu


class _BEVPoolV2(Function):
    @staticmethod
    def symbolic(
        g,
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        out_height,
        out_width,
    ):
        return g.op(
            "BEVPoolV2TRT",
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            out_height_i=out_height,
            out_width_i=out_width,
        )

    @staticmethod
    def forward(
        ctx,
        depth,  # N,D,H,W
        feat,  # N,H,W,C
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        out_height,
        out_width,
    ):
        """run forward."""
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        bev_feat_shape = (
            depth.shape[0],
            1,
            out_height,
            out_width,
            feat.shape[-1],
        )  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2_gpu(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
        )
        bev_feat = bev_feat.squeeze(2)
        bev_feat = bev_feat.permute(0, 2, 3, 1)
        return bev_feat

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class _BEVPoolV2_2(_BEVPoolV2):
    @staticmethod
    def symbolic(
        g,
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        out_height,
        out_width,
    ):
        return g.op(
            "BEVPoolV2TRT2",
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            out_height_i=out_height,
            out_width_i=out_width,
        )


_bev_pool_v2 = _BEVPoolV2.apply
_bev_pool_v2_2 = _BEVPoolV2_2.apply


def bev_pool_v2(
    depth,  # N,D,H,W
    feat,  # N,H,W,C
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_starts,
    interval_lengths,
    out_height=128,
    out_width=128,
):
    return _bev_pool_v2(
        depth,  # N,D,H,W
        feat,  # N,H,W,C
        ranks_depth.int(),
        ranks_feat.int(),
        ranks_bev.int(),
        interval_starts.int(),
        interval_lengths.int(),
        out_height,
        out_width,
    )


def bev_pool_v2_2(
    depth,  # N,D,H,W
    feat,  # N,H,W,C
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_starts,
    interval_lengths,
    out_height=128,
    out_width=128,
):
    return _bev_pool_v2_2(
        depth,  # N,D,H,W
        feat,  # N,H,W,C
        ranks_depth.int(),
        ranks_feat.int(),
        ranks_bev.int(),
        interval_starts.int(),
        interval_lengths.int(),
        out_height,
        out_width,
    )
