import numpy as np
from collections import deque
from math import pi, log10, sin, tanh
from broyden_solver import householder_like_lu_colored
import networkx as nx
import matplotlib.pyplot as plt
import pickle


# ==============================
# Simulation Configuration
# ==============================
N = 990
E = 991
seed_num = 70

#BFS and Multi source min distance functions
def multi_source_min_dist(adj, sources):
    """
    Minimum of edge numbers from multiple starting nodes
    (sources) at once
    """
    dist = {}
    q = deque()
    for s in sources:
        dist[int(s)] = 0
        q.append(int(s))
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

#Distance from a specific node
def bfs_dist(adj, start):
    """
    Shortest graph distance from one start node
    to all reachable nodes
    """
    start = int(start)
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

#Adding the pair of edges
def edge_key(i, j):
    """
    Builds graph pairs
    """
    i = int(i); j = int(j)
    return (i, j) if i < j else (j, i)


#Assigning source & demand
def generate_source_demand_nodes(N, seed_num=68):
    """
    Generating source, demand and internal nodes with specific characteristics.
    """
    rng = np.random.default_rng(seed_num)
    #The network is just too small
    if N < 3:
        raise ValueError("N must be greater than 3!")

    N_tot = np.arange(1, N+1)
    #The network is small
    if 3 <= N < 10:
        #Just 1 source and demand
        N_s = 1
        N_d = 1
        if N >= 4:
            N_d = 2
        N_internal = N - N_s - N_d
    else:
        #total_bd ~ Total source and demand nodes
        total_bd = int(0.2 * N)
        total_bd = max(total_bd, 3)
        if total_bd >= N:
            total_bd = N - 1
        #max_Ns is the maximum number of sources which is lesser than demands for sure
        max_Ns = (total_bd - 1) // 2
        if max_Ns < 1:
            N_s = 1
            N_d = 2
        else:
            N_s = int(rng.integers(1, max_Ns + 1))
            N_d = int(total_bd - N_s)

        N_internal = N - N_s - N_d

    sources = rng.choice(np.arange(1, N+1), size=N_s, replace=False)
    remains = list(set(range(1, N+1)) - set(sources))
    demands = rng.choice(remains, size=N_d, replace=False)
    internals = list(set(range(1, N+1)) - set(sources) - set(demands))
    #Mapping the pressure of the source nodes
    P_sou_map = {int(i): float(rng.uniform(3e6, 6e6)) for i in sources}

    return N_tot, N_s, N_d, N_internal, [int(x) for x in sources], [int(x) for x in demands], [int(x) for x in internals], P_sou_map


def generate_pipe_network_loopsafe(
    N, num_pipes, seed_num=68,
    min_loop_ratio=0.25,          # Minimm number of loops in the network where E >= (N-1) + ceil(min_loop_ratio*N)
    deg_max=10,                   # Max degree of connectivity 
    min_hops_source_to_demand=2,  # demands at last 2 edges away from sources
    min_hops_between_sources=2,   # sources are also 2 edges away from each other
    max_graph_tries=25,           # Max tries till we find a graph
    max_pick_tries=2000           # Max tries in order to find sources, demands and internals
):
    """
    Generating the desired network which satisfied the conditions.
    """
    rng = np.random.default_rng(seed_num)

    if N < 5:
        raise ValueError("N must be >= 5")

    #Adding loops to the number of edges
    min_loops = int(np.ceil(min_loop_ratio * N))
    min_E = (N - 1) + min_loops

    # Every node can connect to (N-1) other nodes and also we want unique edge(So we divided by 2)
    cap_E = N * (N - 1) // 2

    # enforce enough loops but respect cap
    E_target = int(max(min_E, min(num_pipes, cap_E)))
    E_target = min(E_target, cap_E)
    # We are try building the graph up to max_graph_tries(If the generated graph did not satisfied conditions we will rebuild it! )
    for gtry in range(max_graph_tries):
        # Spanning tree starting from a random node(Constructive method)
        nodes = np.arange(1, N + 1)
        rng.shuffle(nodes)
        # Initialization
        edges = []
        degree = {int(i): 0 for i in range(1, N + 1)}
        adj = {int(i): [] for i in range(1, N + 1)}
        # Loop to connect new random nodes to already connected node
        for k in range(1, N):
            j = int(nodes[k])
            # Forces to pick one of the nodes which is already connected
            i = int(rng.choice(nodes[:k]))
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1
            adj[i].append(j)
            adj[j].append(i)

        edge_set = set(edge_key(i, j) for (i, j) in edges)

        # Add extra edges up to E_target with the consideration of deg_cap
        tries = 0
        max_add_tries = 150 * max(1, E_target)

        while len(edges) < E_target and tries < max_add_tries:
            tries += 1
            i = int(rng.integers(1, N + 1))
            j = int(rng.integers(1, N + 1))
            # node_i != node_j
            if i == j:
                continue
            # degree of node_i should not be higher than deg_max
            if degree[i] >= deg_max or degree[j] >= deg_max:
                continue
            kkey = edge_key(i, j)
            # if the edge is already in edge_set ignore this and choose another one
            if kkey in edge_set:
                continue

            edge_set.add(kkey)
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1
            adj[i].append(j)
            adj[j].append(i)

        # Function to pick sources/demands/internal
        # Tries random assignments until it finds a valid one
        def pick_sets():
            N_tot, N_s, N_d, N_internal, _, _, _, _ = generate_source_demand_nodes(
                N, seed_num + 1000 + gtry
            )
            #Picking different nodes as source and demand until the requirements has been met
            for _ in range(max_pick_tries):
                perm = rng.permutation(np.arange(1, N + 1))
                sources   = [int(x) for x in perm[:N_s]]
                demands   = [int(x) for x in perm[N_s:N_s + N_d]]
                internals = [int(x) for x in perm[N_s + N_d:]]

                # If source directly connected to a demand node reject the graph
                bad_sd = False
                dem_set = set(demands)
                for s in sources:
                    for nb in adj[s]:
                        if nb in dem_set:
                            bad_sd = True
                            break
                    if bad_sd:
                        break
                if bad_sd:
                    continue

                # Using Multi_source_min_dist function that I wrote earlier in order to check the minimum distance of demands from sources
                dist_min = multi_source_min_dist(adj, sources)
                if any(dist_min.get(int(d), 10**9) < min_hops_source_to_demand for d in demands):  #get(int(d), 10**9) just in case if d is missing it does not cause crash
                    continue

                # Checking if the sources have enough distance between them!
                if len(sources) > 1 and min_hops_between_sources > 0:
                    ok = True
                    for i in range(len(sources)):
                        #Finding the distance from source i
                        di = bfs_dist(adj, sources[i])
                        for j in range(i + 1, len(sources)):
                            #Checking the other sources' distance from source i
                            if di.get(sources[j], 10**9) < min_hops_between_sources:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue

                # Building the pressure mapping for sources
                P_sou_map = {int(s): float(rng.uniform(3e6, 6e6)) for s in sources}
                # If graph has built with in max_pick tries, returns Number of sources, demands, internal and the nodes themselves and also pressure map
                return N_tot, N_s, N_d, N_internal, sources, demands, internals, P_sou_map
            # If it did not succeed, return nothing
            return None

        # If picked results failed raise an Error, else continue with the last constraint
        picked = pick_sets()
        if picked is None:
            continue

        N_tot, N_s, N_d, N_internal, sources, demands, internals, P_sou_map = picked

        # Enforce internal degree >= 2 (As they cannot have degree of 0 or 1 in the graph)
        add_tries = 0
        for v in internals:
            v = int(v)
            while degree[v] < 2 and add_tries < 5000:
                add_tries += 1
                cand = [u for u in range(1, N + 1)
                        if u != v and edge_key(v, u) not in edge_set and degree[u] < deg_max]
                #We have not candidates for adding edges
                if not cand:
                    break
                #Randomly choose from candidate to add a edge to node v which has degree <= 2
                u = int(rng.choice(cand))

                edge_set.add(edge_key(v, u))
                edges.append((v, u))
                degree[v] += 1
                degree[u] += 1
                adj[v].append(u)
                adj[u].append(v)
        #Building connectivity matrix, map and the number of edges  that we have used!
        connectivity_matrix = [(int(i), int(j)) for (i, j) in edges]
        connectivity_map = {int(i): [int(x) for x in adj[int(i)]] for i in range(1, N + 1)}
        E_used = len(connectivity_matrix)

        return connectivity_map, connectivity_matrix, E_used, N_tot, N_s, N_d, N_internal, sources, demands, internals, P_sou_map

    raise RuntimeError("Failed to build a loopsafe network. Increase num_pipes or relax constraints.")

#Generating physical properties based on the assignment(The pump fraction identifies the number of pumps which are presented on the graph)
def generate_physical_prop(connectivity_matrix, seed_num=68, pump_fraction=0.15):
    rng = np.random.default_rng(seed_num)

    E = len(connectivity_matrix)
    A = []
    # Physical ranges (Note that I run the problem first with decreasing ranges in physical properties, and later increased the range, 
    # I have realized that alpha's range cannot be more than (-0.03, 0.03)). As a consequence, I later implement homotopy method in order that I could solve the problem
    # for alpha in the range of (-0.1, 0.1)

    D     = rng.uniform(0.3, 1, size=E)
    #D     = rng.uniform(0.3, 0.7, size=E)      
    #D     = rng.uniform(0.3, 1, size=E)       
    eps   = rng.uniform(1e-6, 1e-3, size=E)     
    #eps   = rng.uniform(1e-6, 1e-3, size=E)    
    L = rng.uniform(100.0, 3000.0, size=E)
    #L     = rng.uniform(100.0, 900.0, size=E)  
    #L     = rng.uniform(100.0, 3000.0, size=E) 
    #alpha = rng.uniform(-0.025, 0.025, size=E)
    alpha = rng.uniform(-0.03, 0.03, size=E)
    #alpha = rng.uniform(-0.1, 0.1, size=E)    

    # Pump fraction(Assigning on which node we have pumps and adding the flag for that node)
    P_flag = np.zeros(E, dtype=int)
    n_pumps = max(1, int(pump_fraction * E))
    pump_idx = rng.choice(E, size=n_pumps, replace=False)
    P_flag[pump_idx] = 1

    # Building the final connectivity matrix with physical properties for every edge
    for k, (i, j) in enumerate(connectivity_matrix):
        tuple_k = (
            int(i),
            int(j),
            float(D[k]),
            float(eps[k]),
            float(L[k]),
            float(alpha[k]),
            int(P_flag[k])
        )
        A.append(tuple_k)

    return A

def generate_physical_prop_for_edges(edges, seed_num=68, pump_fraction=0.0):
    rng = np.random.default_rng(seed_num)
    E = len(edges)
    A = []

    D     = rng.uniform(0.3, 1, size=E)
    eps   = rng.uniform(1e-6, 1e-3, size=E)
    L     = rng.uniform(100.0, 3000.0, size=E)
    alpha = rng.uniform(-0.03, 0.03, size=E)

    P_flag = np.zeros(E, dtype=int)
    n_pumps = max(1, int(pump_fraction * E)) if E > 0 else 0
    if n_pumps > 0:
        pump_idx = rng.choice(E, size=n_pumps, replace=False)
        P_flag[pump_idx] = 1

    for k, (i, j) in enumerate(edges):
        A.append((
            int(i),
            int(j),
            float(D[k]),
            float(eps[k]),
            float(L[k]),
            float(alpha[k]),
            int(P_flag[k])
        ))

    return A

def add_edges_to_existing(connectivity_matrix, N, E_target, seed_num=68, deg_max=10):
    rng = np.random.default_rng(seed_num)
    edges = list((int(i), int(j)) for (i, j) in connectivity_matrix)
    edge_set = set(edge_key(i, j) for (i, j) in edges)

    degree = {int(i): 0 for i in range(1, N + 1)}
    adj = {int(i): [] for i in range(1, N + 1)}
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1
        adj[i].append(j)
        adj[j].append(i)

    tries = 0
    max_add_tries = 150 * max(1, E_target)
    new_edges = []

    while len(edges) < E_target and tries < max_add_tries:
        tries += 1
        i = int(rng.integers(1, N + 1))
        j = int(rng.integers(1, N + 1))
        if i == j:
            continue
        if degree[i] >= deg_max or degree[j] >= deg_max:
            continue
        kkey = edge_key(i, j)
        if kkey in edge_set:
            continue

        edge_set.add(kkey)
        edges.append((i, j))
        new_edges.append((i, j))
        degree[i] += 1
        degree[j] += 1
        adj[i].append(j)
        adj[j].append(i)

    connectivity_map = {int(i): [int(x) for x in adj[int(i)]] for i in range(1, N + 1)}
    return connectivity_map, edges, new_edges

#Mass balance equation and mapping dict for internal nodes function
def mass_balance_indicies(N_tot, internals, sources, demands, N_internal, A):
    #  In mapping_dict: internal nodes are mapped to index in x[:N_internal] and fixed nodes(sources/demands) are mapped to -1 (We later use this for building the residuals
    mapping_dict = {}
    index = 0
    internals_set = set(int(v) for v in internals)
    sources_set   = set(int(v) for v in sources)
    demands_set   = set(int(v) for v in demands)

    for v in N_tot:
        v = int(v)
        if v in internals_set:
            mapping_dict[v] = index
            index += 1
        elif (v in sources_set) or (v in demands_set):
            mapping_dict[v] = -1

    # Building Q_in / Q_out for internal nodes only
    # x = [p_internal, Q_edges] so edge k is x[N_internal + k]
    Q_out = {int(n): [] for n in internals_set}
    Q_in  = {int(n): [] for n in internals_set}

    # One pass over edges
    for k, pipe in enumerate(A):
        i = int(pipe[0])
        j = int(pipe[1])

        q_idx = N_internal + k
        # Index j means the flow has entered to node j, index i means the flow has left node i
        if i in internals_set:
            Q_out[i].append(q_idx)
        if j in internals_set:
            Q_in[j].append(q_idx)
    #Returning Q_in, Q_out and mapping_dict in order to build the residuals
    return Q_in, Q_out, mapping_dict

# Function that builds the residuals
def network_func(x, N_internal, internals, sources, Q_in, Q_out, mapping_dict, P_sou_map,
                 A, E, ro, miu, g, alpha_p, P_dem,
                 H0=45.0, q_switch=1e-3, demands=None,
                 neighbor_penalty_w=0.0, neighbor_penalty_eps=1.0, neighbor_penalty_margin=0.0,
                 report_pumps=False):
    """
    Makes the residuals for Broyden solver(Continuity and Energy balance equations)
    """

    internals_set = set(int(v) for v in internals)
    sources_set   = set(int(v) for v in sources)
    demands_set   = set(int(v) for v in (demands or []))

  
    # Continuity residuals 
    # First N_internal equations are continuity equation residuals

    F = np.zeros(N_internal + E, dtype=float)
    for m, n in enumerate(internals):
        n = int(n)
        # for n'th node in internals finding Q_in indecies that enters the node, then find which variable correspond to this Q_in in x vector!(Doing the same thing for Q_out)
        F[m] = sum(x[q_idx] for q_idx in Q_in[n]) - sum(x[q_idx] for q_idx in Q_out[n])

    # Energy residuals
    # We have as many energy residuals as the number of edges in the network, they will start from index N_internal0 till N_internal+k. k is the number of edges.
    pump_total = 0

    for k, pipe in enumerate(A):
        i, j, D, eps, L, alpha, p_flag = (
            int(pipe[0]), int(pipe[1]),
            float(pipe[2]), float(pipe[3]), float(pipe[4]), float(pipe[5]),
            float(pipe[6])
        )

        # We use mapping_dict and P_sou_map that we built previously in order to assign the pressure of internal nodes and source nodes(for both i and j nodes)
        if i in internals_set:
            P_i = x[mapping_dict[i]]
        elif i in sources_set:
            P_i = P_sou_map[i]
        else:
            P_i = P_dem

        if j in internals_set:
            P_j = x[mapping_dict[j]]
        elif j in sources_set:
            P_j = P_sou_map[j]
        else:
            P_j = P_dem
        # Q_internal variable in x vector as the first N_internal elements are pressure variables
        Qk = x[N_internal + k]

        # Cross-sectional area
        A_cs = 0.25 * pi * D**2

        # Reynolds
        Re = (4.0 * ro * abs(Qk)) / (pi * D * miu + 1e-30)

        # Friction factor 
        if Re < 2100.0:
            f = 64.0 / max(Re, 1e-30) # If Re ever reduced to 0 gaurding that
        else:
            term = (eps/(3.7*D))**1.11 + 6.9/max(Re, 1e-30)
            f = 1.0 / (-1.8 * log10(term))**2

        # We should use Q|Q| instead of Q^2 in order to consider the flow direction in head loss or else head loss would always be positive
        h_loss = f * (L/D) * (Qk * abs(Qk)) / (2.0 * g * A_cs**2 + 1e-30)

        # You had delta_z = L*sin(alpha) and then subtracted it in residual.
        delta_z = L * sin(alpha)

        # Pump equation also changes base on the flow directions.
        #  based on the tanh function since it is a smooth function(tanh)
        # If Qk is large and pos tanh → +1, If Qk is large and neg → -1
        # If Qk near 0, near 0
        # q_switch controls the transition. smaller q_switch → sharper transition, larger q_switch → smoother transition
        Hp = max(0.0, H0 - alpha_p * Qk**2)
        direction = tanh(Qk / q_switch)
        pump_term = p_flag * ro * g * Hp * direction
        if p_flag != 0:
            pump_total += 1

        # Demand-neighbor soft penalty
        penalty = 0.0
        if neighbor_penalty_w > 0.0 and demands_set:
            if i in demands_set and j not in demands_set:
                xgap = (P_dem - neighbor_penalty_margin) - P_j
                soft = 0.5 * (xgap + np.sqrt(xgap * xgap + neighbor_penalty_eps))
                penalty = neighbor_penalty_w * soft
                penalty = +penalty
            elif j in demands_set and i not in demands_set:
                xgap = (P_dem - neighbor_penalty_margin) - P_i
                soft = 0.5 * (xgap + np.sqrt(xgap * xgap + neighbor_penalty_eps))
                penalty = neighbor_penalty_w * soft
                penalty = -penalty

        #Making energy residuals based on everything
        # Energy residuals 
        F[N_internal + k] = (P_i - P_j
                             - ro * g * (h_loss + delta_z)
                             + pump_term
                             + penalty)

    if report_pumps:
        print(f"Pump terms counted (all flagged pumps): {pump_total}/{pump_total}")

    return F


# Scaling Pressures and Flow-rates based on appropirate scales(I have found these based on the residual amounts)
P_s = 1e6
Q_s = 1e-2

#Scaling the residual function
def F_s(y,
        N_internal, internals, sources, Q_in, Q_out, mapping_dict, P_sou_map,
        A, E, ro, miu, g, alpha_p, P_dem,
        H0=45.0, q_switch=1e-3, demands=None,
        neighbor_penalty_w=0.0, neighbor_penalty_eps=1.0, neighbor_penalty_margin=0.0):
    """
    Scaling the residuals in order that 
    the Broyden solver sees the true values
    """
    # Unscale unknowns
    # y = [P_scaled_internal, Q_scaled_edges]
    P_scaled = y[:N_internal]
    Q_scaled = y[N_internal:]

    P = P_scaled * P_s
    Q = Q_scaled * Q_s

    x = np.concatenate([P, Q])


    # Raw residuals
    F = network_func(
        x, N_internal, internals, sources, Q_in, Q_out, mapping_dict, P_sou_map,
        A, E, ro, miu, g, alpha_p, P_dem,
        H0=H0, q_switch=q_switch, demands=demands,
        neighbor_penalty_w=neighbor_penalty_w, neighbor_penalty_eps=neighbor_penalty_eps,
        neighbor_penalty_margin=neighbor_penalty_margin
    )

    # Scale equations
    # continuity: divide by Q_s
    # energy:      divide by P_s
    F_cont  = F[:N_internal] / Q_s
    F_energ = F[N_internal:] / P_s

    return np.concatenate([F_cont, F_energ])

#ADDING Grouping funcs for Jacobian coloring
def build_rows_by_col(N_internal, internals, mapping_dict, Q_in, Q_out, A, E):
    """
    Finding a specific column like(j) affects which row?
    """
    n = N_internal + E
    rows_by_col = [set() for _ in range(n)]

    # continuity row index depends on internals LIST order (important!)
    #Mapping internal nodes to their continuity equations(They are in order)
    cont_row_of_node = {int(node): m for m, node in enumerate(internals)}

    internals_set = set(int(v) for v in internals)

   
   
  
    # Looping through the internal nodes(r is just the node continuity equation's index)
    # For each internal node
    for node in internals:
        node = int(node)
        # Find the row index of this node's continuity equation.
        r = cont_row_of_node[node]
        # Every flow that enters this node affects this continuity equation
        for q_idx in Q_in[node]:
            rows_by_col[q_idx].add(r)
        # Every flow that leaves this node also affects this continuity equation
        for q_idx in Q_out[node]:
            rows_by_col[q_idx].add(r)

    # Qk affects the k'th energy equation too
    for k in range(E):
        q_col = N_internal + k
        e_row = N_internal + k
        rows_by_col[q_col].add(e_row)

    # Pressure columns: appear only in energy equations of incident edges
    # for each of k energy equations
    for k, pipe in enumerate(A):
        # Extract Pi and Pj specified to that energy equation
        i = int(pipe[0]); j = int(pipe[1])
        # This is the k'th energy equation
        e_row = N_internal + k
        # If i or j belongs to internal nodes, find their indicies in variable vector and assign e_row to them
        if i in internals_set:
            rows_by_col[mapping_dict[i]].add(e_row)
        if j in internals_set:
            rows_by_col[mapping_dict[j]].add(e_row)

    # Convert to sorted lists for faster loops later
    rows_by_col = [np.array(sorted(list(s)), dtype=int) for s in rows_by_col]
    return rows_by_col


def greedy_color_columns(rows_by_col):
    """
    Creating greedy groups which do not have any conflicts with 
    each other
    """

    # Number of columns
    n = len(rows_by_col)
    # Groups list of lists
    groups = []
    # Tracks which row already taken by each group, to elaborate,
    # Imagine rows by columns is something like this: rows_by_col = col0: {0, 3}, col1: {1}, col2: {3} , col3: {2} ,col4: {0}..col2 and col4 conflicts with
    # col0
    # We save col0 in group_rows and later when we move to col1 we check rows in order that they do not have any conflict
    group_rows = []  

    for col in range(n):
        #  For each column finds equations that will be affected by this column
        rows = rows_by_col[col]
        # try to place in existing group
        placed = False
        # checking already existing groups
        for gi in range(len(groups)):
            # If the variable(column) have not affect anything append it to this group
            if len(rows) == 0:
                groups[gi].append(col)
                placed = True
                break
            # Check if the equations in gi group do not have overlap with equations in rows, if yes append this col to the gi group and update group_rows[gi]
            if group_rows[gi].isdisjoint(rows):
                groups[gi].append(col)
                group_rows[gi].update(rows.tolist())
                placed = True
                break
        # If equations in rows have overlapped with equations in every group, then start a new group with col
        if not placed:
            groups.append([col])
            group_rows.append(set(rows.tolist()))

    # In the end return groups
    return groups



# User define properties

min_loop_ratio = 0.1
pump_fraction = 0.15

tol = 1e-7
max_outer_tries = 8
verbose = True


# Physical constants
ro = 998.2
miu = 1.002e-3
g = 9.80665
alpha_p = 500.0
P_dem = 1e6

neighbor_penalty_w = 2e4
neighbor_penalty_eps = 1e4
neighbor_penalty_margin = 0.0
penalty_schedule = [2e4, 5e4, 1e5, 2e5, 3e5, 5e5, 9e5]
penalty_scale = 1.0


# Scaling
P_s = 1e6
Q_s = 1e-2


# Retry / loop boost settings
E_step_ratio = 0.1
E_try = int(E)
prev_y_star = None
prev_E = None

# 1) build graph + base solve; if base fails or negative pressures, add edges
ok_base = False
ok = False
y_base = None
F_base = None
iters_base = 0
y_star = None
F_star = None
iters = 0

for trial in range(1, max_outer_tries + 1):

    # 1) build graph on first try; then add loops to existing graph
    if trial == 1:
        (connectivity_map, connectivity_matrix, E_used,
         N_tot, N_s, N_d, N_internal,
         sources, demands, internals, P_sou_map) = generate_pipe_network_loopsafe(
            N, E_try,
            seed_num=seed_num + (trial - 1),
            min_loop_ratio=min_loop_ratio,
            deg_max=10,
            min_hops_source_to_demand=2,
            min_hops_between_sources=2,
            max_graph_tries=25,
            max_pick_tries=2000
        )

        # 2) assign pipe properties
        A = generate_physical_prop(
            connectivity_matrix,
            seed_num=seed_num + (trial - 1),
            pump_fraction=pump_fraction
        )
    else:
        connectivity_map, connectivity_matrix, new_edges = add_edges_to_existing(
            connectivity_matrix,
            N,
            E_try,
            seed_num=seed_num + (trial - 1),
            deg_max=10
        )
        if new_edges:
            new_A = generate_physical_prop_for_edges(
                new_edges,
                seed_num=seed_num + 5000 + (trial - 1),
                pump_fraction=pump_fraction
            )
            A = list(A) + list(new_A)
        E_used = len(A)

    # 3) continuity bookkeeping
    Q_in, Q_out, mapping_dict = mass_balance_indicies(
        N_tot, internals, sources, demands, N_internal, A
    )

    # 4) sparsity pattern + greedy coloring
    rows_by_col = build_rows_by_col(
        N_internal, internals, mapping_dict, Q_in, Q_out, A, E_used
    )
    groups = greedy_color_columns(rows_by_col)

    if verbose:
        print("\n" + "-" * 64)
        print(f"[Trial {trial:02d}] Target edges: {E_try} | Used: {E_used}")
        print(f"[Trial {trial:02d}] System size: Variables={N_internal + E_used}, Equations={N_internal + E_used}")
        print(f"[Trial {trial:02d}] Graph details: N={N}, E={E_used}, Loops={E_used - (N - 1)}")
        print(f"[Trial {trial:02d}] Color groups: {len(groups)}")
        print(f"[Trial {trial:02d}] Boundary details: Sources={len(sources)} Demands={len(demands)}")

    # 5) initial guess (scaled)
    if prev_y_star is not None and prev_E is not None and E_used >= prev_E:
        n_new = E_used - prev_E
        q_new = np.full(n_new, 1.0e-2 / Q_s, dtype=float)
        y0 = np.concatenate([prev_y_star[:N_internal], prev_y_star[N_internal:], q_new])
    else:
        p_int0 = np.full(N_internal, 2.0e6, dtype=float)   # Pa
        Q0     = np.full(E_used,     1.0e-2, dtype=float)  # m^3/s
        y0 = np.concatenate([p_int0 / P_s, Q0 / Q_s])

    # 6) solver wrapper (penalty weight is argument)
    def solve_with_penalty(y_init, w):
        def F_scaled(y):
            return F_s(
                y,
                N_internal, internals, sources, Q_in, Q_out, mapping_dict, P_sou_map,
                A, E_used, ro, miu, g, alpha_p, P_dem,
                H0=45.0, q_switch=1e-3, demands=demands,
                neighbor_penalty_w=w, neighbor_penalty_eps=neighbor_penalty_eps,
                neighbor_penalty_margin=neighbor_penalty_margin
            )
        return householder_like_lu_colored(
            F_scaled, y_init,
            rows_by_col=rows_by_col, groups=groups,
            eps=tol,
            max_outer_num=600, max_inner_num=25,
            rel_step=1e-7,
            verbose=verbose
        )

    # 7) base solve (w = 0)
    if verbose:
        print(f"[Trial {trial:02d}] Base solve (w = 0.00e+00)")
    y_base, F_base, ok_base, iters_base = solve_with_penalty(y0, 0.0)

    # base pressure positivity check
    if ok_base:
        p_internal_base = y_base[:N_internal] * P_s
        p_full_base = np.zeros(N, dtype=float)
        for s in sources:
            p_full_base[s - 1] = P_sou_map[s]
        for d in demands:
            p_full_base[d - 1] = P_dem
        for v in internals:
            p_full_base[v - 1] = p_internal_base[mapping_dict[int(v)]]
        negp_base = int(np.sum(p_full_base < 0))
    else:
        negp_base = 1

    if (not ok_base) or (negp_base > 0):
        print(f"[retry] Base solve failed or negative pressures (negp={negp_base}). Adding edges.")
        prev_y_star = y_base if ok_base else None
        prev_E = E_used if ok_base else None
        bump = int(max(10, E_step_ratio * N))
        E_try_old = E_try
        E_try = int(E_try + bump)
        print(f"[retry] Increasing target edges: E {E_try_old} -> {E_try}")
        continue

    # 8) penalty continuation with retries (reduce schedule if it fails)
    y_star, F_star, ok, iters = y_base, F_base, ok_base, iters_base
    for ptry in range(1, max_outer_tries + 1):
        if verbose:
            print(f"[Penalty Trial {ptry:02d}] scale={penalty_scale:.3f}")
        ok = True
        for w in [w0 * penalty_scale for w0 in penalty_schedule]:
            if verbose:
                print(f"[Penalty Trial {ptry:02d}] w = {w:.2e}")
            y_star, F_star, ok, iters = solve_with_penalty(y_star, w)
            if not ok:
                break
        if ok:
            break
        penalty_scale *= 0.5
        y_star = y_base

    if ok:
        break

    # if penalty still failed after reductions, try adding edges
    print("[retry] Penalty continuation failed; adding edges.")
    prev_y_star = y_base
    prev_E = E_used
    bump = int(max(10, E_step_ratio * N))
    E_try_old = E_try
    E_try = int(E_try + bump)
    print(f"[retry] Increasing target edges: E {E_try_old} -> {E_try}")

# Fallback if all trials failed to reach penalty stage
if y_star is None:
    y_star = y_base
    F_star = F_base
    ok = False
    iters = iters_base

# 9) unscale results
p_internal = y_star[:N_internal] * P_s
Q_sol      = y_star[N_internal:] * Q_s

# 10) expand full pressures for reporting
p_full = np.zeros(N, dtype=float)
for s in sources:
    p_full[s - 1] = P_sou_map[s]      # 1-based -> index
for d in demands:
    p_full[d - 1] = P_dem
for v in internals:
    p_full[v - 1] = p_internal[mapping_dict[int(v)]]

# 11) Calculate metrics
Fn = float(L2norm(F_star))
negp = int(np.sum(p_full < 0))

pmin_bar = float(p_full.min() / 1e5)
pmax_bar = float(p_full.max() / 1e5)
qmin = float(Q_sol.min())
qmax = float(Q_sol.max())
pump_total = int(sum(int(pipe[6]) for pipe in A))

# 12) Detailed summary
print("\n" + "=" * 64)
print("Summary")
print("=" * 64)
print(f"Convergence status: {'YES' if ok else 'NO'}")
print(f"Iterations: {iters}")
print(f"Residual norm (||F||): {Fn:.4e}")
print(f"Pressure range: {pmin_bar:.2f} .. {pmax_bar:.2f} bar   (Negative pressures: {negp})")
print(f"Flow range: {qmin:.4e} .. {qmax:.4e} m^3/s")
print(f"Total edges used: {E_used}   |   Additional loops = {E_used - (N - 1)}")
print(f"Total pumps (flagged): {pump_total}")

# 13) Store output
out = dict(
    connectivity_map=connectivity_map,
    connectivity_matrix=connectivity_matrix,
    A=A,
    N_tot=N_tot,
    N_s=N_s, N_d=N_d, N_internal=N_internal,
    sources=sources, demands=demands, internals=internals,
    P_sou_map=P_sou_map,
    p_full=p_full,
    p_internal=p_internal,
    Q_sol=Q_sol,
    F_sol=F_star,
    ok=ok,
    iters=iters
)


# Final report after all trials
print("\n" + "#" * 64)
print("FINAL REPORT")
print("#" * 64)
print(f"Converged: {out['ok']}")
print(f"Total iterations: {out['iters']}")
print(f"Final ||F||2: {float(L2norm(out['F_sol'])):.6e}")
print(f"Pressure range (bar): {float(out['p_full'].min()/1e5):.3f} .. {float(out['p_full'].max()/1e5):.3f}")
print(f"Flow range (m^3/s): {float(out['Q_sol'].min()):.6e} .. {float(out['Q_sol'].max()):.6e}")
print(f"Negative pressures: {int(np.sum(out['p_full'] < 0))}")
print(f"Sources (n={len(out['sources'])}): {sorted(out['sources'])[:15]}{' ...' if len(out['sources'])>15 else ''}")
print(f"Demands (n={len(out['demands'])}): {sorted(out['demands'])[:15]}{' ...' if len(out['demands'])>15 else ''}")
print("#" * 64)

# Save baseline (w=0) and with-penalty runs
if ok_base:
    baseline_payload = dict(
        out=out,
        y_star_scaled=y_base,
        y0_scaled=y0,
        N=N,
        E_used=E_used,
        seed_num=seed_num,
        min_loop_ratio=min_loop_ratio,
        pump_fraction=pump_fraction
    )
    BASELINE_PKL = f"baseline_N{N}_E{E_used}_seed{seed_num}.pkl"
    with open(BASELINE_PKL, "wb") as f:
        pickle.dump(baseline_payload, f)
    print(f"[saved] Baseline run -> {BASELINE_PKL}")
else:
    print("[saved] Skipped baseline save (base solve failed).")

if ok:
    with_penalty_payload = dict(
        out=out,
        y_star_scaled=y_star,
        y0_scaled=y0,
        N=N,
        E_used=E_used,
        seed_num=seed_num,
        min_loop_ratio=min_loop_ratio,
        pump_fraction=pump_fraction,
        penalty_schedule=penalty_schedule,
        penalty_scale=penalty_scale
    )
    WITH_PENALTY_PKL = f"baseline_with_penalty_N{N}_E{E_used}_seed{seed_num}.pkl"
    with open(WITH_PENALTY_PKL, "wb") as f:
        pickle.dump(with_penalty_payload, f)
    print(f"[saved] With-penalty run -> {WITH_PENALTY_PKL}")
else:
    print("[saved] Skipped with-penalty save (penalized solve failed).")

# Plotting the network that has been solved
def build_graph_from_connectivity(connectivity_map, connectivity_matrix):
    """
    If connectivity_map looks like {node: [nbrs]}, uses it.
    Else falls back to connectivity_matrix (dense or sparse).
    Nodes are assumed to be 1-based IDs: 1..N
    """
    G = nx.Graph()

    # Try connectivity_map first
    if isinstance(connectivity_map, dict) and len(connectivity_map) > 0:
        # adjacency dict: {u: [v1,v2,...]}
        for u, nbrs in connectivity_map.items():
            if nbrs is None:
                continue
            for v in nbrs:
                if u != v:
                    G.add_edge(int(u), int(v))
        if G.number_of_edges() > 0:
            return G

    # Otherwise, use connectivity_matrix
    M = connectivity_matrix
    if hasattr(M, "tocoo"):  # sparse
        coo = M.tocoo()
        for i, j, val in zip(coo.row, coo.col, coo.data):
            if val != 0 and i < j:
                G.add_edge(int(i + 1), int(j + 1))
    else:  # dense numpy
        idx = np.transpose(np.nonzero(np.triu(M, 1)))
        for i, j in idx:
            G.add_edge(int(i + 1), int(j + 1))

    return G

# Build the final graph from the accepted output 
G = build_graph_from_connectivity(out["connectivity_map"], out["connectivity_matrix"])

# Layout (increase spread by using a larger k and scale)
# k controls ideal distance; larger -> more spread. scale expands final positions.
k_layout = 6.0 / np.sqrt(G.number_of_nodes())
pos = nx.spring_layout(G, seed=seed_num, iterations=160, k=k_layout, scale=8.0)

#  Node coloring by pressure (bar)
p_bar = out["p_full"] / 1e5  # bar, indexed 0..N-1
node_color = np.array([p_bar[n - 1] for n in G.nodes()], dtype=float)

plt.figure(figsize=(11, 9))

# Edges 
nx.draw_networkx_edges(G, pos, alpha=0.1, width=1, edge_color = "black")

# All nodes colored by pressure
sc = nx.draw_networkx_nodes(
    G, pos,
    node_size=18,
    node_color=node_color,
    cmap="viridis",
    linewidths=0
)

# Highlight sources and demands on top
nx.draw_networkx_nodes(G, pos, nodelist=out["sources"], node_size=40, node_color="red", label="Sources")
nx.draw_networkx_nodes(G, pos, nodelist=out["demands"], node_size=40, node_color="orange", label="Demands")

cb = plt.colorbar(sc)
cb.set_label("Pressure [bar]")

F_norm = float(L2norm(out["F_sol"]))
plt.title(f"Solved network (N={G.number_of_nodes()}, E={G.number_of_edges()}, ||F||={F_norm:.2e})")
plt.axis("off")
plt.legend(loc="upper right", markerscale=1.5, frameon=True)
plt.tight_layout()
plt.show()

# Pressure on nodes that are the neighbors of demand nodes(And have pressure less than demand nodes! )
# Neighbors of demand nodes with pressure < demand pressure
demands = list(out["demands"])
p_full  = out["p_full"]  # Pa, index 0..N-1

print("\n" + "-" * 64)
print("DEMAND-NEIGHBORS WITH PRESSURE LOWER THAN THE DEMAND NODE")
print("-" * 64)

total_hits = 0
neighbor_tol_bar = 0.011
neighbor_tol_pa = neighbor_tol_bar * 1e5

for d in sorted(demands):
    P_d = float(p_full[d - 1])  # Pa at demand node

    # collect neighbors with lower pressure
    low_nbrs = []
    for n in G.neighbors(d):
        P_n = float(p_full[n - 1])
        if P_n < P_d - neighbor_tol_pa:
            low_nbrs.append((n, P_n))

    if not low_nbrs:
        continue

    # sort by "how low" they are (most below demand first)
    low_nbrs.sort(key=lambda t: t[1])  # ascending pressure

    total_hits += len(low_nbrs)

    print(f"\nDemand node {d}: P_dem = {P_d/1e5:.3f} bar ({P_d:.3e} Pa)")
    print("  Neighbor   P[bar]      dP = (P_n - P_dem) [bar]")
    print("  -------   --------    ---------------------------")
    for n, P_n in low_nbrs:
        dP_bar = (P_n - P_d) / 1e5
        print(f"  {n:7d}   {P_n/1e5:8.3f}    {dP_bar: .5f}")

print("\n" + "-" * 64)
print(f"Total neighbors found with P_neighbor < P_demand: {total_hits}")
print("-" * 64)
