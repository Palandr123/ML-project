from typing import Optional
import math

import torch
import torchvision.transforms.functional as TF
import numpy as np
import einops


def calculate_weighted_centroid(
    attention_sum: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the weighted centroid of a set of coordinates based on attention weights.

    Args:
        attention_sum: torch.Tensor - tensor representing the sum of attention weights along either width or height.
            It should have shape (n, 1, k), where n is the batch size, and k is the number of attention values.
        coordinates: torch.Tensor - tensor containing the coordinates to be used for the centroid calculation.
            It should have shape (n, h or w, k), where n is the batch size, and h or w represents the height
            or width of the coordinates, and k is the number of coordinates.

    Returns:
        torch.Tensor - tensor representing the weighted centroid of the coordinates based on the attention weights.
            It has shape (n, k), where n is the batch size, and k is the number of coordinates.
    """
    # Normalize attention weights and calculate weighted centroid
    normalized_attention = attention_sum / (
        attention_sum.sum(-2, keepdim=True) + 1e-4
    )  # (n, 1, k)
    weighted_centroid = (
        coordinates[None, ..., None] * normalized_attention
    )  # (n, h or w, k)
    return weighted_centroid.sum(-2)  # (n, k)


def centroid(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate the centroid of attention weights.

    Args:
        a: torch.Tensor - input attention tensor of shape (n, h, w, k).

    Returns:
        torch.Tensor - the centroid of the attention in the form of (n, k, 2).
    """
    n, h, w, k = attention_weights.shape

    # Calculate 1D vectors for x and y coordinates
    x = torch.linspace(0, 1, w, device=attention_weights.device)
    y = torch.linspace(0, 1, h, device=attention_weights.device)

    # Calculate the sum of attention weights along the width and height dimensions
    sum_along_width = attention_weights.sum(-3)  # (n, w, k)
    sum_along_height = attention_weights.sum(-2)  # (n, h, k)

    # Calculate centroids along the x and y directions
    centroid_x = calculate_weighted_centroid(sum_along_width, x)
    centroid_y = calculate_weighted_centroid(sum_along_height, y)
    centroid = torch.stack((centroid_x, centroid_y), -1)  # (n, k, 2)
    return centroid


def normalized_attention(
    report_attention: torch.Tensor,
    hard: bool = False,
    threshold: float = 0.5,
    scaling_factor: float = 10,
) -> torch.Tensor:
    """
    Compute the normalized attention.

    Args:
        report_attention: torch.Tensor - input attention tensor.
        hard: bool - if True, use a hard threshold, else use a sigmoid-based threshold.
        threshold: float - threshold value.
        scaling_factor: float - scaling factor.

    Returns:
        torch.Tensor - normalized attention difference
    """
    min_attention = report_attention.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    max_attention = report_attention.max(2, keepdim=True)[0].max(3, keepdim=True)[0]

    # Compute attention values relative to the min and max (with a small epsilon to avoid division by zero)
    attention_thresholded = (report_attention - min_attention) / (
        max_attention - min_attention + 1e-4
    )

    if hard:
        # Apply a hard threshold and convert to a float tensor (0 or 1)
        return (attention_thresholded > threshold).float()

    attention_binarized = torch.sigmoid(
        (attention_thresholded - threshold) * scaling_factor
    )
    min_binarized = attention_binarized.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    max_binarized = attention_binarized.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    attention_normalized = (attention_binarized - min_binarized) / (
        max_binarized - min_binarized + 1e-4
    )
    return attention_normalized


def compute_appearance(
    last_feats: torch.Tensor, last_attn: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Compute the appearance based on feature maps, attention maps, and a threshold.

    Args:
        last_feats: torch.Tensor - a tensor representing the feature maps. The shape should be (batch_size, height, width, channels).
        last_attn: torch.Tensor - a tensor representing the attention maps. The shape should be (batch_size, height, width).
        threshold: float - a threshold value used for attention normalization.

    Returns:
        torch.Tensor: A tensor representing the computed appearance. The shape is (batch_size, channels).
    """
    last_attn = last_attn.detach()
    last_feats = last_feats.permute(0, 2, 3, 1)

    last_attn = normalized_attention(last_attn, hard=True, threshold=threshold)
    last_attn = TF.resize(
        last_attn.permute(0, 3, 1, 2), last_feats.shape[1], antialias=True
    ).permute(0, 2, 3, 1)

    appearance = (last_attn[..., None] * last_feats[..., None, :]).sum((-3, -4)) / (
        1e-4 + last_attn.sum((-2, -3))[..., None]
    )

    return appearance


def appearance_difference(
    aux: dict[str, dict[str, Optional[torch.Tensor]]],
    i: int,
    tgt: dict[str, dict[str, Optional[torch.Tensor]]],
    idxs: Optional[np.ndarray[np.int64]] = None,
    L2: bool = False,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Calculate appearance difference based on auxiliary information.

    Args:
        aux: dict[str, dict[str, Optional[torch.Tensor]]] - dictionary containing the source features and attention maps.
        i: int - index of the target element.
        tgt: dict[str, dict[str, Optional[torch.Tensor]]] - dictionary containing the target features and attention maps.
        idxs: Optional[np.ndarray[np.int64]] - indexes to select specific components.
        L2: bool - if True, calculate L2 distance, else calculate L1.
        threshold: float - threshold value for attention binarization.

    Returns:
        torch.Tensor - Appearance difference.
    """
    target_aux = tgt
    device = torch.utils._pytree.tree_flatten(aux)[0][-1].device

    # Extract relevant components from auxiliary information
    aux = {k: next(iter(v.values()))[i] for k, v in aux.items()}
    target_aux = {
        k: next(iter(v.values()))[i].detach().to(device) for k, v in target_aux.items()
    }

    if idxs is not None:
        aux["last_attn"] = aux["last_attn"][..., idxs]
        target_aux["last_attn"] = target_aux["last_attn"][..., idxs]

    original_appearance = compute_appearance(**aux, threshold=threshold)
    target_appearance = compute_appearance(**target_aux, threshold=threshold)

    if L2:
        return (0.5 * (original_appearance - target_appearance) ** 2).mean()
    return (original_appearance - target_appearance).abs().mean()


def silhouette_difference(
    attn: torch.Tensor,
    i: int,
    tgt: torch.Tensor,
    idxs: Optional[np.ndarray[np.int64]] = None,
    rot: float = 0.0,
    sy: float = 1.0,
    sx: float = 1.0,
    dy: float = 0.0,
    dx: float = 0.0,
    threshold: bool = True,
    resize: Optional[int] = None,
    L2: bool = False,
) -> torch.Tensor:
    """
    Calculate silhouette difference between attention distributions.
    Args:
        attn: torch.Tensor - input attention tensor.
        i: int - index of the target element.
        tgt: torch.Tensor - target attention tensor.
        idxs: Optional[np.ndarray[np.int64]] - indexes to select specific attention components.
        rot: float - rotation angle in degrees.
        sy: float - scaling factor along the y-axis.
        sx: float - scaling factor along the x-axis.
        dy: float - vertical shift.
        dx: float - horizontal shift.
        threshold: bool - if True, use thresholding, else use raw attention values.
        resize: int - resize dimension.
        L2: bool - if True, calculate L2 distance, else calculate absolute difference.

    Returns:
        torch.Tensor: Silhouette difference.
    """
    attn = attn[i]
    tgt_attn = tgt[i].to(attn.device)

    if idxs is not None:
        attn = attn[..., idxs]
        tgt_attn = tgt_attn[..., idxs]

    if resize:
        attn = TF.resize(attn.permute(0, 3, 1, 2), resize, antialias=True).permute(
            0, 2, 3, 1
        )
        tgt_attn = TF.resize(
            tgt_attn.permute(0, 3, 1, 2), resize, antialias=True
        ).permute(0, 2, 3, 1)

    if threshold:
        attn = normalized_attention(attn)
        tgt_attn = normalized_attention(tgt_attn, hard=True)

    requires_transform = rot != 0 or any(_ != 1.0 for _ in [sy, sx, dy, dx])
    if requires_transform:
        ns, hs, ws, ks = tgt_attn.shape
        dev = attn.device
        n, h, w, k = torch.meshgrid(
            torch.arange(ns),
            torch.arange(ws),
            torch.arange(hs),
            torch.arange(ks),
            indexing="ij",
        )
        n, h, w, k = n.to(dev), h.to(dev), w.to(dev), k.to(dev)
        # centroid
        c = centroid(attn)
        ch = c[..., 1][:, None, None] * hs
        cw = c[..., 0][:, None, None] * ws
        # object centric coord system
        h = h - ch
        w = w - cw
        # rotate
        angle_deg_cw = rot
        th = angle_deg_cw * math.pi / 180
        wh = torch.stack((w, h), -1)[..., None]
        R = torch.tensor(
            [[math.cos(th), math.sin(th)], [math.sin(-th), math.cos(th)]]
        ).to(dev)
        wh = (R @ wh)[..., 0]
        w = wh[..., 0]
        h = wh[..., 1]
        # resize
        h = h / sy
        w = w / sx
        # shift
        y_shift = dy * hs * sy
        x_shift = dx * ws * sx
        h = h - y_shift
        w = w - x_shift
        h = h + ch
        w = w + cw

        h_normalized = (2 * h / (hs - 1)) - 1
        w_normalized = (2 * w / (ws - 1)) - 1
        coords = torch.stack((w_normalized, h_normalized), dim=-1)
        coords_unnorm = torch.stack((w, h), dim=-1)

        coords = coords[:, :, :, 0, :]
        coords_unnorm = coords_unnorm[:, :, :, 0, :]

        # Collapse the batch_size, num_tokens dimension and set num_channels=1 for grid sampling
        tgt_attn = einops.rearrange(tgt_attn, "n h w k -> n k h w")
        tgt_attn = torch.nn.functional.grid_sample(
            tgt_attn, coords, mode="bilinear", align_corners=False
        )
        tgt_attn = einops.rearrange(tgt_attn, "n k h w -> n h w k")
    if L2:
        return (0.5 * (attn - tgt_attn) ** 2).mean()
    return (attn - tgt_attn).abs().mean()


def change_centroid(
    attn: torch.Tensor,
    i: int,
    tgt: Optional[torch.Tensor] = None,
    shift: tuple[int, int] = (0.0, 0.0),
    relative: bool = False,
    idxs: Optional[np.ndarray[np.int64]] = None,
    L2: bool = False,
) -> torch.Tensor:
    """
    Compute and return the change in centroid between 'attn' and 'tgt' tensors for the specified
    dimension 'i'. The centroid is calculated based on the attention scores.

    Args:
        attn: torch.Tensor - the input attention tensor.
        i: int - the dimension along which to calculate the centroid.
        tgt: Optional[torch.Tensor] - the target attention tensor. Defaults to None.
        shift: tuple[int, int] - the shift to apply to the centroid, specified as a tuple of
                                 two integers. Defaults to (0.0, 0.0).
        relative: bool - if True, calculate the centroid relative to the target. Defaults to False.
        idxs: Optional[np.ndarray[np.int64]] - An array of indices to consider for the centroid
                                               calculation. Defaults to None.
        L2: bool - if True, calculate the L2 distance between centroids. Defaults to False.

    Returns: 
        torch.Tensor - the change in centroid, which can be either the absolute mean difference or the
        mean squared difference, depending on the value of 'L2'.
    """
    attn = attn[i]
    tgt_attn = tgt[i].to(attn.device) if tgt is not None else None

    if relative:
        assert tgt_attn is not None
    tgt_attn = tgt_attn if tgt_attn is not None else attn
    if idxs is not None:
        attn = attn[..., idxs]
        tgt_attn = tgt_attn[..., idxs]
    shift = torch.tensor(shift).to(attn.device)

    obs_centroid = centroid(attn)
    tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)
    if relative:
        tgt_centroid = centroid(tgt_attn) + tgt_centroid
    if L2:
        return (0.5 * (obs_centroid - tgt_centroid) ** 2).mean()
    return (obs_centroid - tgt_centroid).abs().mean()


def size(attn: torch.Tensor, thresh: bool = True) -> torch.Tensor:
    """
    Compute the size of attention weights in a tensor.

    Args:
        attn: torch.Tensor - a tensor representing attention weights, typically obtained from an attention mechanism.
        thresh: bool - if True (default), the attention weights are normalized before computing the size. 
                       If False, the original attention weights are used.

    Returns:
    torch.Tensor - a tensor containing the size of the attention weights. The size is computed as the mean of the attention weights 
                   along dimensions -2 and -3. The result is reshaped to have an additional dimension.
    """
    if thresh:
        attn_norm = normalized_attention(attn)
    else:
        attn_norm = attn.clone()
    return attn_norm.mean((-2, -3))[..., None]


def change_size(
    attn: torch.Tensor,
    i: int,
    tgt: Optional[torch.Tensor] = None,
    relative: bool = False,
    shift: tuple[int] = (0.0,),
    thresh: bool = True,
    idxs: Optional[np.ndarray[np.int64]] = None,
    L2: bool = False,
) -> torch.Tensor:
    """
    Change the size of the attention distribution with respect to a target distribution.

    Args:
        attn: torch.Tensor - the input attention distribution.
        i: int - the index specifying which element to operate on within the attention distribution.
        tgt: Optional[torch.Tensor]) - the target attention distribution (optional).
        relative: bool - whether to perform a relative size change (default is False).
        shift: tuple[int] - the shift values for adjusting the size (default is (0.0,)).
        thresh: bool - whether to apply a attention normalization for size calculation (default is True).
        idxs: Optional[np.ndarray[np.int64]] - indices to consider when calculating size (optional).
        L2: bool - whether to use the L2 loss metric for size change (default is False).

    Returns:
        torch.Tensor - the size change of the attention distribution with respect to the target, computed as the mean absolute 
                       difference or mean squared difference (L2 loss), depending on the 'L2' parameter.
    """
    attn = attn[i]
    tgt_attn = tgt[i].to(attn.device) if tgt is not None else None

    if relative:
        assert tgt_attn is not None
    tgt_attn = tgt_attn if tgt_attn is not None else attn
    if idxs is not None:
        attn = attn[..., idxs]
        tgt_attn = tgt_attn[..., idxs]
    shift = torch.tensor(shift).to(attn.device)

    size_obs = size(attn, thresh)
    size_tgt = shift.reshape((1,) * (size_obs.ndim - shift.ndim) + shift.shape)
    if relative:
        size_tgt = size(tgt_attn, thresh) + size_tgt
    if L2:
        return (0.5 * (size_obs - size_tgt) ** 2).mean()
    return (size_obs - size_tgt).abs().mean()
