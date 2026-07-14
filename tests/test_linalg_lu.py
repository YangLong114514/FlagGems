import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_lu
@pytest.mark.parametrize("shape", [(4, 4), (32, 32), (16, 32), (64, 32), (128, 16, 16)])
@pytest.mark.parametrize("pivot", [True, False])
def test_linalg_lu(shape, pivot):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_p, ref_l, ref_u = torch.linalg.lu(ref_inp, pivot=pivot)
    with flag_gems.use_gems():
        res_p, res_l, res_u = torch.linalg.lu(inp, pivot=pivot)

    k = min(inp.shape[-2], inp.shape[-1])

    if pivot:
        # With partial pivoting, the decomposition is stable and unique.
        # Compare P, L, U directly against the reference.
        utils.gems_assert_close(res_l, ref_l, torch.float32)
        utils.gems_assert_close(res_u, ref_u, torch.float32)
        utils.gems_assert_close(res_p, ref_p, torch.float32)
        # Disable TF32 for the reconstruction matmul to avoid precision loss
        _tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            reconstructed = res_p @ res_l @ res_u
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _tf32
        utils.gems_assert_close(
            reconstructed, ref_inp, torch.float32, reduce_dim=k
        )
    else:
        # Without pivoting, LU decomposition can be numerically unstable:
        # L and U factors can differ between implementations when near-zero
        # pivots are encountered. Compare the reconstructed matrices L@U instead
        # of comparing against the original A directly.
        assert res_p.numel() == 0
        assert ref_p.numel() == 0
        _tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            gems_reconstructed = res_l @ res_u
            ref_reconstructed = ref_l @ ref_u
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _tf32
        utils.gems_assert_close(
            gems_reconstructed, ref_reconstructed, torch.float32, atol=1e-2 * k
        )
