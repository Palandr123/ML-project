import torch


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
