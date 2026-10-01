import numpy as np

def num_jacobe_colored(f, y, F0, rows_by_col, groups, rel_step=1e-7):
    """
    Using rows_by_col and groups in order to compute
    Jacobian in a faster way!
    """
    y = np.asarray(y, float)
    F0 = np.asarray(F0, float)
    n = y.size
    J = np.zeros((n, n), dtype=float)
    # Finite difference step size per variable
    # y[j] is large, step is larger
    # y[j] is ~ 0, step is rel_step
    h = rel_step * (1.0 + np.abs(y))
    
    # Loop over each group
    # Each group is a list of column indicies such that for any two
    # columns in the same group, rows_by_col do not overlap.
    for grp in groups:
        y1 = y.copy()
        # As a result, we can perturb all columns in that specific group at once
        # doing many at once in order to save function calls.. Ha Ha Ha! :), Am I not amazing??
        for j in grp:
            y1[j] += h[j]

        F1 = np.asarray(f(y1), float)
        dF = F1 - F0
        
        # This might seem I see multiple variable effects and multiple columns contribute
        # but the key is next loop when I seperate the result based on specific column..HeeeHeee :)
        # For a particular column j, only certain rows can change when j changes, none of the other columns in the group affect
        # those same rows
        for j in grp:
            # Finding rows that will be affected by column j(Since we have done coloring before, all other columns' effects are zero)
            rows = rows_by_col[j]
            if rows.size:
                # For these rows extract the jacobian value
                J[rows, j] = dF[rows] / h[j]

    return J
