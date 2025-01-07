# ------------------------------- #
# Everything needed for qudit LHVs
# ------------------------------- #

# Packages
from typing import Tuple, Callable, Generator
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad, Array
from jax.lax import complex, select, scan, fori_loop, dynamic_slice
from jax.nn import softmax
from optax import apply_updates
from functools import partial, reduce
from itertools import permutations as itertools_permutations


# ---------------- #
# Measurement Rule #
# ---------------- #

# generate d-tuples of non-negative integers summing to S
def sums(
            d: int, 
            S: int
        )   -> Generator[Tuple[int], int, None]:
    
    if d == 1:
        yield (S,)

    else:
        for v in range(S + 1):
            for x in sums(d - 1, S - v):
                yield (v,) + x


# construct d-tuples of non-negative integers summing to S <= D
def construct_powers(
                        D: int,
                        d: int
                    )   -> Array:

    pow = [d*(0,)]
    for S in range(1, D+1):
        for x in sums(d, S): 
            pow.append(x)

    return jnp.array(pow, dtype=jnp.int32)


# construct the n (= (D+d) choose d) monomials, sorted by degree, up to degree D in d variables
def construct_monomials(
                            D: int, 
                            d: int
                        )   -> Tuple[Callable[[ArrayLike], Array], int]: 

    pow = construct_powers(D, d)
    funcs = jit( lambda x, j: jnp.prod(x**pow[j]) ) # monomials
    funcs = vmap( funcs, in_axes=(None, 0) )
    n = pow.shape[0] # number of monomials 

    return jit( lambda x: funcs(x, jnp.arange(n)) ), n # vector valued function, e.g. for d=1: x -> (1, x, x^2, ..., x^D)


# Gram-Schmidt for monomials
def orthonormalize_monomials(
                                X:          ArrayLike, 
                                monomials:  Callable[[ArrayLike], Array], 
                                n:          int, 
                                alpha:      float = 1e-3, 
                                beta:       float = 1e-2
                            )   -> Array:
    
    '''
        Gram-Schmidt decomposition for d-variable polynomials represented by coefficient lists
        polynomials: f_j(x_1,...,x_d) = sum_k coeffs_jk * x_1^(p_1[k]) * ... * x_d^(p_d[k])
        monomials: outputs vector of all monomials (x_1^(p_1[k]) * ... * x_d^(p_d[k]))_(k=1)^n for given input x
        k=1...n indexes all combinations of powers up to total maximal degree D
        Monte Carlo integration over X = matrix of many d-tuples of samples x from the underlying space
    '''                            
    
    def poly(x, c): # construct polynomial from coefficients c and input x
        return jnp.dot(c, monomials(x))
    poly = vmap(poly, in_axes=(0, None))
    func = lambda c: poly(X, c) # Evaluate polynomial defined by coefficients c on all samples x in X

    def L2_dot(u, v): # Monte Carlo estimate for L2 inner product between polynomials given by coefficient lists u and v
        return jnp.mean(func(u)*func(v))
    
    def L2_norm(u): # Monte Carlo estimate for L2 norm (squared) of polynomial given by coefficient list u
        return jnp.mean(func(u)**2)

    def projection(u, v): # project v onto u
        norm2 = L2_norm(u)
        norm2 = jnp.where(norm2>0., norm2, 1.)
        return (L2_dot(u, v) / norm2) * u

    def orthogonalize(v, C_sub): # subtract projections onto vectors in C_sub from v
        projections = vmap(lambda u: projection(u, v))(C_sub)
        return v - jnp.sum(projections, axis=0)
    
    def step_func(j, C_step): # take the jth vector in V (a monomial), orthogonalize wrt. previously obtained vecotors, normalize and add to C
        v = jnp.squeeze(dynamic_slice(V, (j, 0), (1, n)))
        v = orthogonalize(v, C_step)
        norm = jnp.sqrt(L2_norm(v))
        v = jnp.select([norm > alpha], [v/norm], jnp.zeros(n)) # Only add to C if norm of v large enough, otherwise add zeros which get removed later
        return C_step.at[j].set(v)
    
    def clean_up(v): # Set small coefficients to zero, then re-normalize
        w = jnp.abs(v)
        mask = w >= jnp.max(w)*beta
        u = mask*v
        norm = jnp.sqrt(L2_norm(u))
        return u/norm

    V = jnp.eye(n) # corresponds to the monomials
    C = jnp.zeros((n, n)) # filled with coefficient lists corresponding to orthonormalized monomials

    v = V[0] # The first vector only needs to be normalized -> first entry of C
    v /= jnp.sqrt(L2_norm(v))
    C = C.at[0].set(v)
    
    C = fori_loop(1, n, step_func, C) # Loop over monomials
    
    D = []
    for v in C: # Remove the zeros (all coefficients = 0)
        if not jnp.all(v==0.): D.append(v)
    D = jnp.array(D)
    
    return vmap(clean_up)(D)


# permutations of delta elements
def permutations(delta: int) -> Array:  
    
    return jnp.array(list(itertools_permutations(range(delta))))

@vmap
def inverse_permutations(P: ArrayLike) -> Array: # inverse permutation of permutation P
    
    delta = len(P)
    P_inv = jnp.empty_like(P)
    
    return P_inv.at[P].set(jnp.arange(delta))


# Build LHV measurment rule q_k(x, lambda)
def LHV_rule_constructor(
                            delta:              int,                          # number of measurment outcomes
                            n_vars:             int,                          # number of parameters specifying a measurement x
                            D:                  int,                          # maximal polynomial degree for functions B_m in expansion of f
                            params_extractor:   Callable[[ArrayLike], Array], # measurement -> array of n_vars real parameters specifying the measurement (input to B_m)
                            symmetric:          bool = True,                  # whether measurement rule is symmetrized with respect to outcome permutations
                            ONB:                bool = False,                 # whether monomials are orthonormalized
                            coeffs:             ArrayLike | None = None,      # (hidden_dim, number of monomials) = expansion coefficients of B_m into monomials
                                                                                # ignored if ONB==False, automatically constructed if passed as "None"
                            samples:            ArrayLike | None = None,      # batch of measurements used in Monte Carlo integration for orthonormalization of monomials
                            alpha:              float = 1e-4, 
                            beta:               float = 1e-3
                        )   -> Tuple[   Callable[[ArrayLike, ArrayLike], Array],    # measurement rule q(measurement, hidden_variable)
                                        int,                                        # hidden-variable dimension (hidden_dim) per measurement outcome
                                        Array ]:                                    # coeffs (to save or reuse)

    monomials, hidden_dim = construct_monomials(D, n_vars) # monomials in parameters

    if not ONB: # use (all) monomials

        print("Monomials")
        
        # logits f for a batch of measurement-parameters and hidden_variables ((d, hidden_dim)-matrices)
        def logits(measure, hidden_variable):
            return jnp.matmul(hidden_variable, monomials(measure))

    else: # orthonomralize monomials via Gram-Schmidt
        
        print("ONB")

        if coeffs is None: # construct coeffs
            measures_params = vmap(params_extractor)(samples)
            coeffs = orthonormalize_monomials(measures_params, monomials, hidden_dim, alpha=alpha, beta=beta) # Orthonormalize monomials over space of measurements
        
        hidden_dim = coeffs.shape[0]
    
        # logits f for a batch of measurement-parameters and hidden_variables ((d, hidden_dim)-matrices)
        def logits(measure, hidden_variable):
            basis_funcs = jnp.matmul(coeffs, monomials(measure))
            return jnp.matmul(hidden_variable, basis_funcs)
    
    if symmetric: # permutation invariant measurement rule

        print("Symmetric")

        logits = vmap(logits) # vectorize over permutations
        perms = permutations(delta) # create all permutations of delta elements
        perms_inverse = inverse_permutations(perms) # and the inverse permutations
        
        def rule(measure, hidden_variable): # measurement rule q
            measure_perms = measure[perms] # permutations of measurement
            measure_params = vmap(params_extractor)(measure_perms) # extract parameters
            hvs = hidden_variable[perms_inverse] # inverse permutations of hidden-variable matrices
            F = logits(measure_params, hvs) # logits corresponding to all permutations
            return softmax(jnp.mean(F, axis=0)) # measurement probability given by softmax of mean f over all permutations
    
    else: # non-permutation invariant measurement rule
        
        print("Non-Symmetric")

        def rule(measure, hidden_variable): # measurement rule q
            measure_params = params_extractor(measure) # extract parameters
            f = logits(measure_params, hidden_variable) # logits
            return softmax(f) # measurement probability given by softmax of f
    
    print("hidden_dim =", hidden_dim)
    if not (coeffs is None): print("Total number of monomials: ", coeffs.shape[1])

    return rule, hidden_dim, coeffs



# --------------------- #
# Sampling Measurements #
# --------------------- #

# Gram-Schmidt decomposition of vectors in C^d
def gram_schmidt(M: ArrayLike) -> Array: 

    d = M.shape[0]
    U = jnp.zeros_like(M)

    def projection(u, v): # project v onto u
        norm2 = jnp.dot(jnp.conjugate(u), u)
        norm2 = jnp.where(norm2>0., norm2, 1.)
        return (jnp.dot(jnp.conjugate(u), v) / norm2) * u

    def orthogonalize(v, U): # subtract projections onto vectors in U from v
        projections = vmap(lambda u: projection(u, v))(U)
        return v - jnp.sum(projections, axis=0)
    
    def step_func(j, U): # take the jth vector in M, orthogonalize wrt. previously obtained vecotors, normalize and add to U
        v = jnp.squeeze(dynamic_slice(M, (j, 0), (1, d)))
        v = orthogonalize(v, U)
        v /= jnp.linalg.norm(v)
        return U.at[j].set(v)

    v = M[0] # The first vector only needs to be normalized -> first entry of U
    v /= jnp.linalg.norm(v)
    U = U.at[0].set(v)

    return fori_loop(1, d, step_func, U)

gram_schmidt_v = jit(vmap(gram_schmidt)) # vectorize over batch of matrices


# Sample Haar random unitaries by orthonormalizing the rows of random matrices with gaussian entries
def sample_unitaries(
                        key:    ArrayLike,  # PRNG key
                        N:      int,        # number of unitaries to sample
                        d:      int         # dimension
                    )   -> Array:       # (N, d, d) = batch of unitaries
     
    a, b = random.normal(key, (2, N, d, d))
    
    return gram_schmidt_v(a + complex(0., 1.)*b)


# Sample non-degenerate projective measurements as unitaries
def sample_PVMs(
                    key:        ArrayLike,  # PRNG key
                    N_measures: int,        # number of measurements ("batch size")
                    N:          int,        # number of qudits
                    d:          int         # qudit dimension = delta = number of measurement outcomes
                )   -> Array:       # (N_measures, N, d, d) = batch of N-tuples of unitaries
    
    U = sample_unitaries(key, N_measures*N, d)
    
    return jnp.reshape(U, (N_measures, N, d, d))


@vmap
def projectors(v: ArrayLike) -> Array: # projector onto unit vector v
    
    return jnp.outer(jnp.conjugate(v), v)

# gell_mann expansion coefficients of a rank 1 projection matrix P
def gell_mann_rank1(P: ArrayLike) -> Array: 
    
    d = P.shape[0]
    # real part of uppper triangular part gives x type coefficients, imaginary part of lower triangular part gives y type coefficients
    xy = (jnp.real(jnp.triu(P, k=1)) + jnp.imag(jnp.triu(P.T, k=1).T)) * jnp.sqrt(2.*d)
    D = (jnp.tri(d, d, 0) + jnp.diag(-jnp.arange(1, d), k=1))[:-1] # create diagonals of z type Gell-Mann matrices
    l = jnp.arange(1,d)
    s = jnp.sqrt(2./(l*(l+1.)))
    D = (D.T*s).T # correct prefactors
    Gz = vmap(jnp.diag)(D) # z type Gell-Mann matrices
    z = jnp.real(vmap(jnp.trace)(Gz@P)) * jnp.sqrt(d/2.) # z type coefficients
    z = jnp.concatenate([z, jnp.zeros(1)]) 
    xyz = xy+jnp.diag(z) # all d^2-1 coefficients as matrix (the coefficient for the identity is not needed)
    
    return jnp.ravel(xyz)[:-1] # Vector of expansion coefficients

@jit # concatenated gell mann vectors from projectors onto rows of unitary U
def gell_mann_params_PVM(U: ArrayLike) -> Array: 
    
    P = projectors(U[:-1]) # projector onto last row given by normalization -> drop
    G = vmap(gell_mann_rank1)(P) # gell mann vectors for each (rank 1) projection
    
    return jnp.ravel(G) # for qudits of dimension d: n_vars = (d-1)*(d**2-1) parameters per measurement given by unitary U


@vmap # return U@M@U^dagger where M is diagonal matrix with diagonal=diag
def basis_change(U: ArrayLike, diag: ArrayLike) -> Array:

    return jnp.matmul(U*diag, jnp.conjugate(U.T))

# Sample single POVM for d-dimensional qudit
def sample_POVM(
                    key:    ArrayLike,  # PRNG key
                    d:      int         # qudit dimension
                )   -> Array:       # POVM = d^2 measurement operators summing to 1

    key_spectra, key_unitaries = random.split(key)

    spectra = random.uniform(key_spectra, (d**2, d-1), minval=0., maxval=1.) # spectra between 0 and 1
    spectra = jnp.append(spectra, jnp.zeros((d**2,1)), axis=-1) # last eigenvalue = 0
    unitaries = sample_unitaries(key_unitaries, d**2, d) # unitaries for haar random basis change
    M = basis_change(unitaries, spectra) # unnormalized POVM = list of positive-semidefinite matrices
    
    # normalization
    eta = jnp.sum(M, axis=0) 
    evals, evecs = jnp.linalg.eigh(eta)
    evals = 1./jnp.sqrt(evals)
    eta_sqrti = (evecs*evals)@jnp.conjugate(evecs.T)
    
    return jnp.matmul(jnp.matmul(eta_sqrti, M), eta_sqrti)

# Sample POVMs
def sample_POVMs(
                    key:        ArrayLike,  # PRNG key 
                    N_measures: int,        # number of measurements ("batchsize") 
                    N:          int,        # number of qudits
                    d:          int         # qudit dimension
                )   -> Array:       # batch of POVMs (N_measures, N, d^2, d, d)
    
    keys = jnp.array(random.split(key, (N_measures*N,)))
    measures = vmap(sample_POVM, in_axes=(0, None))(keys, d)
    
    return jnp.reshape(measures, (N_measures, N, d**2, d, d))


# gell_mann expansion coefficients of a hermitian matrix
def gell_mann(A: ArrayLike) -> Array:

    d = A.shape[0]
    # real part of uppper triangular part gives x type coefficients, imaginary part of lower triangular part gives y type coefficients
    xy = (jnp.real(jnp.triu(A, k=1)) + jnp.imag(jnp.triu(A.T, k=1).T)) * jnp.sqrt(2.*d)
    D = (jnp.tri(d, d, 0) + jnp.diag(-jnp.arange(1, d), k=1)) # create diagonals of z type Gell-Mann matrices
    l = jnp.arange(1,d)
    s = jnp.sqrt(2./(l*(l+1.)))
    s = jnp.concatenate([s, jnp.sqrt(2./d)*jnp.ones(1)])
    D = (D.T*s).T # correct prefactors
    Gz = vmap(jnp.diag)(D) # z type Gell-Mann matrices
    z = jnp.real(vmap(jnp.trace)(Gz@A)) * jnp.sqrt(d/2.) # z type coefficients
    xyz = jnp.ravel(xy+jnp.diag(z)) # all d^2 coefficients as matrix
    shift = jnp.zeros(d**2).at[-1].set(1./d)
    
    return xyz - shift # subtract 1/d from g_0

@jit # concatenated gell mann vectors from POVM measurement
def gell_mann_params_POVM(M: ArrayLike) -> Array:

    G = vmap(gell_mann)(M[:-1]) # gell mann vectors for each measuremnt operator (except last one which is given by normalization)
    
    return jnp.ravel(G) # for qudits of dimension d: n_vars = (d**2-1)*d**2 parameters per measurement given by unitary U


# ------------------- #
# QM measurement rule #
# ------------------- #

# N qudits of dimension d
# rho: (d^N, d^N) density matrix

# Projective measurements = (N, d) array of states = one projector per particle
def PQM_pvm(measures: ArrayLike, rho: ArrayLike) -> Array:
    psi = reduce(jnp.kron, measures) # tensor product of states
    return jnp.abs(jnp.real( jnp.dot(jnp.conjugate(psi), jnp.matmul(rho, psi)) )) # P_QM = <psi|rho|psi>

# PVM measurements = (N, d, d) array of measurement operators
def PQM_povm(measures: ArrayLike, rho: ArrayLike) -> Array:
    M = reduce(jnp.kron, measures) # tensor product of measurement operators
    return jnp.abs(jnp.real( jnp.trace(jnp.matmul(M, rho)) )) # P_QM = Tr(rho*M_1*...*M_N)



# ----------------- #
# Distance measures #
# ----------------- #

# KL-component for individual probabilities pqm, phv (for one outcome)
def KL(pqm: float, phv:float) -> float: 
    return pqm * select( pqm>0., jnp.log(pqm)-jnp.log(phv), 0. )

# KL-divergence between vecors pqm, phv (probabiliti distributions over outcomes)
def distance_KL(pqm: ArrayLike, phv: ArrayLike) -> float: 
    kls = vmap(KL, in_axes=(0, 0))(pqm, phv)
    return jnp.abs(jnp.sum(kls))

# Square difference between two vectors
def distance_L2(pqm: ArrayLike, phv: ArrayLike) -> float:
    return jnp.sum((pqm-phv)**2)

# L1 distance between two vectors
def distance_L1(pqm: ArrayLike, phv: ArrayLike) -> float:
    return jnp.sum(jnp.abs(pqm-phv))



# ------------ #
# Optimization #
# ------------ #

# Construct loss function from LHV and QM measurement rules and a distance measure for a fixed state
def loss_from_rules(
                        LHV_measurement_rule:   Callable[[ArrayLike, ArrayLike], Array], 
                        QM_measurement_rule:    Callable[[ArrayLike, ArrayLike], Array], 
                        distance_measure:       Callable[[ArrayLike, ArrayLike], float], 
                        quantum_state:          ArrayLike, 
                        delta:                  int,    # number of measurement outcomes
                        N:                      int     # number of qudits
                    ) -> Callable[[ArrayLike, ArrayLike], float]: # Loss function
    
    outer = vmap(lambda m: reduce(jnp.kron, m), in_axes=(1,))
    PHV_t = vmap(LHV_measurement_rule, in_axes=(None, 0)) # vectorize over cloud-dimension
    def PHV(measures, cloud): # Average probabilities for all joint outcomes given the hidden-state cloud
        probs = vmap(PHV_t, in_axes=(0, 1))(measures, cloud) # vectorize over particles, probs = (N_h, N, number of outcomes)
        return jnp.mean(outer(probs), axis=0) # outer product over particles --> delta^N joint outcomes, mean over cloud
    
    indices = jnp.reshape(jnp.indices(N*(delta,)), (N, delta**N)).T

    def deviation(measures, cloud): # Deviation between probabilities predicted by LHV and QM, measures = (N,)+(shape of single particle measurement)
        phv = PHV(measures, cloud) # = d^N probabilities
        qm_measures = measures[jnp.arange(N), indices] # list of delta^N lists of product measurements (N, delta)
        pqm = vmap(QM_measurement_rule, in_axes=(0, None))(qm_measures, quantum_state) # = delta^N probabilities
        return distance_measure(pqm, phv)

    def Loss(cloud, batched_measures): # Loss = mean deviation over batch of combinations of measurements
        return jnp.mean(vmap(deviation, in_axes=(0, None))(batched_measures, cloud))
    
    return Loss


@partial(jit, static_argnums=tuple(range(3, 9))) # Evaluate loss on batch of test measurements
def eval_test_loss(
                    key: ArrayLike,                                                 # PRNG key
                    cloud: ArrayLike,                                               # hidden-variable cloud (N_hidden, N_qudits, delta, hidden_dim)
                    quantum_state: ArrayLike,                                       # Quantum state, e.g. a density matrix or correlation tensor
                    d: int,                                                         # qudit dimension
                    N_measures: int,                                                # number of measurements ("batchsize")
                    LHV_measurement_rule: Callable[[ArrayLike, ArrayLike], Array],
                    QM_measurement_rule: Callable[[ArrayLike, ArrayLike], Array], 
                    distance_measure: Callable[[ArrayLike, ArrayLike], float], 
                    sample: Callable[[ArrayLike, int, int, int], Array]
                  ) -> float:       # test loss

    _, N, delta, _ = cloud.shape
    Loss = loss_from_rules(LHV_measurement_rule, QM_measurement_rule, distance_measure, quantum_state, delta, N)
    batched_measures = sample(key, N_measures, N, d)
    
    return Loss(cloud, batched_measures)


# Optimize Qudit LHV Model
@partial(jit, static_argnums=tuple(range(3, 12)))
def autoLHV(

                key:            ArrayLike,  # initial key for pseudo random number generator
                cloud:          ArrayLike,  # (size of hidden-variable cloud, number of particles, number of measuremnt outcomes, hidden-variable dimension per particle and outcome)-tensor 
                                                # = initial hidden-variables cloud, 
                                                # the LHV_measurement_rule recieves a single hidden variable matrix (number of outcomes, hidden_dim) as input
                quantum_state:  ArrayLike,  # variable specifying the quantum state, e.g. represented by a density matrix,
                                                # the QM_measurement_rule recieves this as input along with the measurement options

                d:                  int,    # qudit dimension
                N_measures:         int,    # number of combinations of measurement options the loss function is averaged over per gradient descent step = batchsize
                N_measures_test:    int,    # batchsize for test loss evaluation (once, in the end)
                N_steps:            int,    # total number of gradient descent steps
            
                LHV_measurement_rule:   Callable[[ArrayLike, ArrayLike], Array],    # (measurement option, hidden-variable matrix) -> probabilities for all possible measurement outcomes
                QM_measurement_rule:    Callable[[ArrayLike, ArrayLike], Array],    # (measurement outcome for each particle, quantum state) -> probability for the joint outcome
                distance_measure:       Callable[[ArrayLike, ArrayLike], float],    # (p_qm, p_lhv) -> non-negative number quanitfying the deviation of two probability distributions given as vectors
                sample:                 Callable[[ArrayLike, int, int, int], Array],# (key, N_measures, number of qudits, delta) -> batch of measurements, 
                                                                                        # format needs to be compatible with LHV/QM_measurement_rule
                optimizer:              object                                      # optax optimizer (e.g. "adam") with some learning rate schedule

            ) -> Tuple[ Array,       # optimized hidden-variable cloud
                        Array]:      # loss value progression during training
                        #Array]:     # loss for final cloud evaluated on N_measures_test measurement combinations


    _, N, delta, _ = cloud.shape # number of qudits, number of measurement outcomes
    Loss = loss_from_rules(LHV_measurement_rule, QM_measurement_rule, distance_measure, quantum_state, delta, N) # Construct the loss function

    def gradient_step(train_data, _): # Single stochastic gradient descent step
        key, opt_state, cloud = train_data
        key, key_SGD = random.split(key)
        # Measurement combinations
        batched_measures = sample(key_SGD, N_measures, N, d)
        # SGD
        loss, grads = value_and_grad(Loss, argnums=0)(cloud, batched_measures)
        updates, opt_state = optimizer.update(grads, opt_state, cloud)
        cloud = apply_updates(cloud, updates)
        return [key, opt_state, cloud], loss

    opt_state = optimizer.init(cloud) # The initial parameters are the passed initial hidden-variable cloud
    [key, opt_state, cloud], loss_values = scan(gradient_step, [key, opt_state, cloud], None, length=N_steps) # Training loop

    # Test loss
    #key, key_test = random.split(key)
    #batched_measures = sample(key_test, N_measures_test, N, d)
    #test_loss = Loss(cloud, batched_measures)

    return cloud, loss_values #, test_loss