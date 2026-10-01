#
#
#

import numpy as np
from colored_jacobian import num_jacobe_colored
from norms import L2norm, infinity_norm_mat
from linear_algebra import lu_factor_fast, lu_solve_fast




## Limited-Memory house-holder class 
# Normally we build H_k+1 update as Hk - uv^T/den
# The running cost is outer product O(n^2), Matrix update O(n^2) and memory for saving the whole matrix O(n^2)
# But here in this class first we set a limited memory(m = 12 steps)
# Then what we do is apply the update to the vector(Say g), hence, H_k+1.g = H0.g - sigma(from t=1 till m=12) (ut.vt^(T).g)/dent
# In this way each term needs only a dot product vt^T.g(cost O(n))
# Then a scalar substraction ut.(alpha(equal to vt^T.g)) (cost O(n))
# Also the base solver H0.g = -J^(-1).g (cost O(n^2))
# The whole cost in this way is O(n^2)+O(mn). As a consequence, It is much cheaper than previous cosr 2*O(n^2)!!
class LBroydenH:
    """
    Builds H0 and H in a fast manner
    """
    # Initializing necessary vectors
    def __init__(self, LU, piv, m=12):
        self.LU = np.asarray(LU, dtype=float)
        self.piv = np.asarray(piv, dtype=int)
        self.m = int(m)

        n = self.piv.size
        self.invp = np.empty_like(self.piv)
        self.invp[self.piv] = np.arange(n)

        self.U_list = []  # u vectors
        self.V_list = []  # v vectors
        self.D_list = []  # denom scalars
    # H0g = -J^(-1)g
    # If g = F(x), then H0_apply(g) is "Do one Newton solve using the last Jacobian"
    def H0_apply(self, g):
        return -lu_solve_fast(self.LU, self.piv, g)
    # Computes H0^T.g = -(J^(-T).g)
    # Because we need it for vector v in update H( v = H.apply_T(delta x) )(in update we have delta_xk^T.Hk = (H_k^T.delta_xk)^T=v^T)
    # Logic is as same as lu_solve_transpose_fast func in LU_Decompositions3.py
    def H0T_apply(self, g):
        g = np.asarray(g, dtype=float).ravel()
        n = self.LU.shape[0]
        if g.size != n:
            raise ValueError(f"H0T_apply: g has size {g.size}, expected {n}")

        w = g[self.invp].copy()

        z = np.empty_like(w)
        for i in range(n):
            s = 0.0
            if i > 0:

                s = float(self.LU[:i, i] @ z[:i])
            z[i] = (w[i] - s) / (self.LU[i, i] + 1e-300)


        x = z.copy()
        for i in range(n - 1, -1, -1):
            if i + 1 < n:
            
                x[i] -= float(self.LU[i+1:, i] @ x[i+1:])

        return -x

    # Apply the full limited-memory operator
    # H_k+1.g = H_k.g - sigma (from t=1 till m=12) ut.vt^(T).g/dent
    def apply(self, g):
        g = np.asarray(g, dtype=float).ravel()
        y = self.H0_apply(g)
        for u, v, den in zip(self.U_list, self.V_list, self.D_list):
            y = y - u * ((v @ g) / (den + 1e-300))
        return y
    # Apply operator for the transpose matrix
    # H_k+1^T.g = H_k^T.g - sigma (from t=1 till m=12) vt.ut^T.g/dent
    # Why swap? because in H_k+1 we had (uv^T) taking transpose from it results in vu^T
    def apply_T(self, g):
        g = np.asarray(g, dtype=float).ravel()
        y = self.H0T_apply(g)
        for u, v, den in zip(self.U_list, self.V_list, self.D_list):
            y = y - v * ((u @ g) / (den + 1e-300))
        return y
    # Record and updaate rank-1 correcetion term(-u.v^T/den)
    # Keeps only last m of them
    def update(self, u, v, den):
        self.U_list.append(np.asarray(u, dtype=float).ravel().copy())
        self.V_list.append(np.asarray(v, dtype=float).ravel().copy())
        self.D_list.append(float(den))

        if len(self.U_list) > self.m:
            self.U_list.pop(0)
            self.V_list.pop(0)
            self.D_list.pop(0)


# Newton solver
def householder_like_lu_colored(
    f, x0,rows_by_col, groups,
    eps=1e-8, max_outer_num=50, max_inner_num=20,
    rel_step=1e-7,m_hist=12,verbose=False):
    """
    solving a nonlinear system with an inital guess and residual
    vectors
    """
    # Taking the value of inputs and the function
    x = np.asarray(x0, dtype=float)
    Fk = np.asarray(f(x), dtype=float)

    recomp_J = True
    outer_iter = 0
    Fk_norm = float(L2norm(Fk))

    # Inititalizing LBroydenH operator
    H = None  

    # Jacobian rebuild counter
    rebuild_count = 0
    max_rebuilds = 1

    # Main outer loop that checks the norm of residuals and number of outer iterations
    while (Fk_norm > eps) and (outer_iter < max_outer_num):

        if verbose:
            print(f"Step {outer_iter:3d}  ||RES|| = {Fk_norm:.3e}")

        # Recompute J and reset inverse operator H (If inner loop(line search) failed)
        if recomp_J or outer_iter == 0 or H is None:
            rebuild_count += 1
            if rebuild_count > max_rebuilds:
                if verbose:
                    print(" Jacobian recomputed too many times without convergence!")
                return x, Fk, False, outer_iter

        # Bulding the first step Jacobian matrix and Broyden
            J = num_jacobe_colored(f, x, Fk, rows_by_col, groups, rel_step=rel_step)
            LU, piv = lu_factor_fast(J)
            H = LBroydenH(LU, piv, m=m_hist)
            recomp_J = False

            if verbose:
                print(f" Rebuilding Jacobian matrix! ")
        # Computing fi^2 in order to update the step of line search(s)
        F_k2 = float(Fk @ Fk)

        # step: delta_x = H @ Fk
        delta_x = H.apply(Fk)
        # Initilizing s
        s = 1.0

        # line search(Inner loop of the solver)
        x_k1 = x + s * delta_x
        F_k1 = np.asarray(f(x_k1), dtype=float)
        F_k12 = float(F_k1 @ F_k1)

        inner_iter = 0
        # Checking which s is acceptable based on Broyden update
        while (F_k12 > F_k2) and (inner_iter < max_inner_num):
            eta = F_k12 / (F_k2 + 1e-300)
            s = ((1.0 + 6.0 * eta) ** 0.5 - 1.0) / (3.0 * eta + 1e-300)
            # If s is tiny break the loop
            if s < 1e-6:
                break
            x_k1 = x + s * delta_x
            F_k1 = np.asarray(f(x_k1), dtype=float)
            F_k12 = float(F_k1 @ F_k1)
            inner_iter += 1

        # line search failed -> force Jacobian rebuild
        if F_k12 >= F_k2:
            recomp_J = True
            if verbose:
                print(" Line search did not succeed")
            outer_iter += 1
            continue

        # House-holder update
        # Computing u, v, den for the update
        Y = F_k1 - Fk
        HY = H.apply(Y)

        u = HY + s * delta_x
        v = H.apply_T(delta_x)
        den = float(delta_x @ HY) 
        # If denom is tiny recompute Jacobian
        if abs(den) < 1e-14:
            x, Fk = x_k1, F_k1
            Fk_norm = float(L2norm(Fk))
            recomp_J = True
            if verbose:
                print("  tiny denom; forcing J rebuild next iter.")
            outer_iter += 1
            continue

        H.update(u, v, den)

        # Accept step
        x, Fk = x_k1, F_k1
        Fk_norm = float(L2norm(Fk))
        outer_iter += 1

        # reset rebuild counter on success 
        rebuild_count = 0

        if verbose:
            print(f"  accepted: s={s:.3e}, inner={inner_iter}, denom={den:.3e}")

    return x, Fk, (float(L2norm(Fk)) <= eps), outer_iter
