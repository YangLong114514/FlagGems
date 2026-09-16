import logging

from flag_gems.ops.linalg_norm import _parse_ord, _v_norm

from .linalg_matrix_norm import linalg_matrix_norm
from .vector_norm import vector_norm

logger = logging.getLogger(__name__)


def _matrix_ord_supported(ord):
    """Ascend's linalg_matrix_norm covers the SVD-based ords (2, -2, nuc)
    only; other matrix ords (fro/1/-1/inf/-inf) crash or hang CANN native,
    so linalg_norm rejects them up front instead of delegating."""
    if isinstance(ord, str):
        return ord == "nuc"
    return abs(float(ord)) == 2


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Mirror ``torch.linalg.norm`` dispatch on Ascend.

    Matrix branch (ord="fro"/"nuc", dim as 2-tuple, or 2D input with
    dim=None) reuses the Ascend ``linalg_matrix_norm``; non-SVD matrix ords
    are rejected up front since they crash or hang CANN native.  Vector
    branch reuses the Ascend ``vector_norm``; the per-row p-norm cases run
    the fixed ``_v_norm`` kernel shared with the generic implementation.
    """
    logger.debug("GEMS_ASCEND LINALG_NORM")
    ord = _parse_ord(ord)
    if dim is not None:
        dim = [dim] if isinstance(dim, int) else list(dim)
        if len(dim) not in (1, 2):
            raise RuntimeError(
                f"linalg.norm: If dim is specified, it must be of length 1 or 2. "
                f"Got {dim}."
            )
    elif ord is not None:
        if A.ndim not in (1, 2):
            raise RuntimeError(
                "linalg.norm: If dim is not specified but ord is, "
                f"the input must be 1D or 2D. Got {A.ndim}D."
            )
    if (
        isinstance(ord, str)
        or (dim is not None and len(dim) == 2)
        or (dim is None and A.ndim == 2)
    ):
        ord = "fro" if ord is None else ord
        if not _matrix_ord_supported(ord):
            raise NotImplementedError(
                f"FlagGems Ascend linalg_norm: matrix norm ord '{ord}' is not "
                "supported; Ascend matrix_norm supports 2, -2, nuc only."
            )
        return linalg_matrix_norm(
            A,
            ord,
            (-2, -1) if dim is None else dim,
            keepdim,
            dtype=dtype,
        )
    ord = 2 if ord is None else ord
    if (
        dim is not None
        and len(dim) == 1
        and len(dim) < A.ndim
        and ord not in (2, float("inf"), float("-inf"), 0)
    ):
        return _v_norm(A, ord, dim, keepdim, dtype)
    return vector_norm(A, ord, dim, keepdim, dtype=dtype)
