# src/models/factorized/factor_utils.py

import torch


def permute_dims(
    z
):
    """
    FactorVAE latent permutation.

    Input:
        [B,D]

    Output:
        [B,D]

    Each latent dimension is
    independently shuffled.
    """

    B, D = z.shape

    permuted = []

    for d in range(D):

        idx = torch.randperm(
            B,
            device=z.device
        )

        permuted.append(
            z[idx, d]
        )

    return torch.stack(
        permuted,
        dim=1
    )