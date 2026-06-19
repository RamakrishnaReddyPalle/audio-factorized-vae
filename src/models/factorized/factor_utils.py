# src/models/factorized/factor_utils.py

import torch


def permute_dims(
    z
):
    """
    FactorVAE latent permutation.

    Input:
        z : [B, D]

    Output:
        [B, D]

    Each latent dimension is
    independently shuffled across
    the batch dimension.

    This destroys inter-dimension
    dependencies while preserving
    marginal distributions.
    """

    if z.ndim != 2:

        raise ValueError(

            f"permute_dims expected "
            f"[B,D] tensor, got "
            f"{tuple(z.shape)}"
        )

    batch_size, latent_dim = z.shape

    # ----------------------------------
    # Batch size 1
    #
    # Cannot permute meaningfully.
    #
    # Return clone to avoid accidental
    # in-place interactions later.
    # ----------------------------------

    if batch_size <= 1:

        return z.clone()

    permuted_dims = []

    for d in range(
        latent_dim
    ):

        permutation = torch.randperm(

            batch_size,

            device=z.device
        )

        permuted_dims.append(

            z[
                permutation,
                d
            ]
        )

    return torch.stack(

        permuted_dims,

        dim=1
    )