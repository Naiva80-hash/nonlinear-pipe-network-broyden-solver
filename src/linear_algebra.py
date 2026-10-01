import numpy as np
from norms import infinity_norm_mat
def lu_factor_fast(A, pivot_tol=1e-12):
    LU = np.array(A, dtype=float, copy=True, order="C")
    if LU.ndim != 2 or LU.shape[0] != LU.shape[1]:
        raise ValueError(f"lu_factor_fast expects square 2D, got {LU.shape}")
    n = LU.shape[0]
    piv = np.arange(n)

    scale = infinity_norm_mat(LU)
    tiny = pivot_tol * max(1.0, scale)
    #Iterate over each column k of the matrix, then performing LU factorization in two parts
    for k in range(n):
        # Find row(imax) in the current column k where the row value is maximum absolute
        # If imax is not as same as k, swap kth with imaxth rows in both LU and the pivot vector
        imax = k + np.argmax(np.abs(LU[k:, k]))
        if imax != k:
            LU[[k, imax], :] = LU[[imax, k], :]
            piv[[k, imax]] = piv[[imax, k]]
        # If LU[k, k](The diagonal elemetn) is smaller than the threshold, raise an error indicating the matrix is ill-conditioned or singular
        pivval = LU[k, k]
        if abs(pivval) < tiny:
            raise ValueError(
                f"Singular/ill-conditioned pivot at col {k}: |Ukk|={abs(pivval):.3e}, tiny={tiny:.3e}"
            )

        # multipliers (L column) in-place
        if k + 1 < n:
            # Dividing the elements below the diagonal to pivot element of that column in order to make the diagonal 1 and find Ls elements
            LU[k+1:, k] /= pivval

            # trailing update: A22 -= L21 * U12  (rank-1 update)
            # This is the big speedup: vectorized outer product.
            # It is as same as gauss elimination technically!
            LU[k+1:, k+1:] -= np.outer(LU[k+1:, k], LU[k, k+1:])

    return LU, piv


def lu_solve_fast(LU, piv, b):
    # Multiple b logic
    b = np.asarray(b, dtype=float)
    vec = (b.ndim == 1)
    if vec:
        b = b.reshape(-1, 1)

    n = LU.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"b has shape {b.shape}, expected ({n}, k)")

    # Apply permutation Pb
    # If pivoting has happened to A, it has its affects on b too
    x = b[piv, :].copy()

    # Forward solve: Ly = Pb
    # L is strictly below diag in LU, diag is 1.
    for i in range(n):
        if i > 0:
            x[i, :] -= LU[i, :i] @ x[:i, :]

    # Back solve: U x = y
    for i in range(n-1, -1, -1):
        if i + 1 < n:
            x[i, :] -= LU[i, i+1:] @ x[i+1:, :]
        x[i, :] /= (LU[i, i] + 1e-300)

    return x.ravel() if vec else x


# Solving the transpose system (A^Tx = b) uses the same LU factorization as A(For faster run-time)
def lu_solve_transpose_fast(LU, piv, b):
    b = np.asarray(b, dtype=float)
    vec = (b.ndim == 1)
    if vec:
        b = b.reshape(-1, 1)

    n = LU.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"b has shape {b.shape}, expected ({n}, k)")

    # Building P^(-T).b and taking it as w
    invp = np.empty_like(piv)
    invp[piv] = np.arange(n)
    w = b[invp, :].copy()

    # Solve U^T z = w (U^T is lower)(Taking it from LU factors)
    z = np.zeros_like(w)
    for i in range(n):
        if i > 0:
            z[i, :] = w[i, :] - (LU[:i, i].reshape(1, -1) @ z[:i, :])
        else:
            z[i, :] = w[i, :]
        z[i, :] /= (LU[i, i] + 1e-300)

    # Solve L^T x = z (L^T is upper)
    x = z.copy()
    for i in range(n-1, -1, -1):
        if i + 1 < n:
            # entries of L^T above diag come from LU[j,i] for j>i
            x[i, :] -= (LU[i+1:, i].reshape(1, -1) @ x[i+1:, :])

    return x.ravel() if vec else x
