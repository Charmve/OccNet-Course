import torch


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]

    box_width = bboxes[..., 3:4].log()
    box_length = bboxes[..., 4:5].log()
    box_height = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (
                cx,
                cy,
                box_width,
                box_length,
                cz,
                box_height,
                rot.sin(),
                rot.cos(),
                vx,
                vy,
            ),
            dim=-1,
        )
    else:
        normalized_bboxes = torch.cat(
            (
                cx,
                cy,
                box_width,
                box_length,
                cz,
                box_height,
                rot.sin(),
                rot.cos(),
            ),  # noqa E501
            dim=-1,
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    box_width = normalized_bboxes[..., 2:3]
    box_length = normalized_bboxes[..., 3:4]
    box_height = normalized_bboxes[..., 5:6]

    box_width = box_width.exp()
    box_length = box_length.exp()
    box_height = box_height.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, box_width, box_length, box_height, rot, vx, vy],
            dim=-1,  # noqa E501
        )
    else:
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, box_width, box_length, box_height, rot], dim=-1
        )
    return denormalized_bboxes
