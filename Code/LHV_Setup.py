# Packages 
import jax.numpy as jnp
from jax import jit, vmap, random
from jax.lax import select
from jax.nn import sigmoid
import sympy as sym


#####################
# Distance measures #
#####################

# Return a distance measure between two probabilities pqm = P_QM(up) and phv = P_LHV(up)

# Kulback-Leibler divergence
def KL(pqm, phv):
    return jnp.abs(pqm*select(pqm>0., jnp.log(pqm)-jnp.log(phv), 0.) \
           + (1.-pqm)*select((1.-pqm)>0., jnp.log(1.-pqm)-jnp.log(1.-phv), 0.))

# L2 / squared difference
def L2(pqm, phv):
    return (pqm-phv)**2

# L1 / difference (not suitable for training, but for testing)
def L1(pqm, phv):
    return jnp.abs(pqm-phv)


########################
# QM measurement rules #
########################

# measures: measurement directions for all spins (e.g. (N_spins, 3)-array for 3d projective measurements)
# second parameter specifies the state, e.g. the correlation matrix for general states or the visibility for Werner states
# returns: probability for "all spins up" along the specified measurement directions

# Defines matrix elements for measurement, to be multiplied with the correlation matrix
def measure_product(measures, indices): 
    M = jnp.pad(measures, ((0, 0), (1, 0)), mode='constant', constant_values=1.)
    return jnp.prod(M[jnp.arange(len(indices)), indices])
measure_products = jnp.vectorize(measure_product, excluded=[0], signature="(m)->()")

# General N qubit states represented by all their correlation functions
def PQM(measures, corrs):
    N = measures.shape[0] 
    indices = jnp.array(jnp.meshgrid(*jnp.outer(jnp.repeat(1, N),jnp.arange(4)), indexing='ij')) # all possible correlations (pauli strings)
    indices = jnp.moveaxis(indices, 0, -1)
    M = measure_products(measures, indices)
    return jnp.sum(corrs*M)/2**N

# Two-qubit Werner states parameterized by their visibility v
def PQM_Werner(measures, v):
    return (1. - v*jnp.dot(measures[0], measures[1])) / 4.

# Two-qubit XYZ-states parameterized by their correlators (x,y,z)
def PQM_XYZ(measures, corrs):
    return (1. + jnp.sum(jnp.prod(measures, axis=0)*corrs)) / 4.


#########################
# LHV measurement rules #
#########################

# returns:
# 1) the measurement function: (measurment direction, hidden state vector) --> probability for "up"
# 2) the required hidden state dimension

# Bell's rule ( = spherical harmonics up to degree 1 w/o normalization)
def PLHV_Bell():
    hidden_dim = 3
    def rule(measure, hidden_state):
        return sigmoid(jnp.dot(measure, hidden_state))
    return rule, hidden_dim

# Odd Spherical harmonics up to degree 3
def PLHV_sh3():
    hidden_dim = 3 + 7
    def rule(measure, hidden_state):
        x, y, z = measure
        m_vec = jnp.array( \
            [jnp.sqrt(3.) * x, \
             jnp.sqrt(3.) * y, \
             jnp.sqrt(3.) * z, \
             jnp.sqrt(35./8.)  * (3.*x**2-y**2)*y, \
             jnp.sqrt(105.)    * (x*y*z), \
             jnp.sqrt(21./8.)  * (5.*z**2-1.)*y, \
             jnp.sqrt(7./4.)   * (5.*z**2-3.)*z, \
             jnp.sqrt(21./8.)  * (5.*z**2-1.)*x, \
             jnp.sqrt(105./4.) * (x**2-y**2)*z, \
             jnp.sqrt(35./8.)  * (x**2-3.*y**2)*x ]) / jnp.sqrt(4.*jnp.pi)
        return sigmoid(jnp.dot(m_vec, hidden_state))
    return rule, hidden_dim

# Odd Spherical harmonics up to degree 5
def PLHV_sh5():
    hidden_dim = 3 + 7 + 11
    def rule(measure, hidden_state):
        x, y, z = measure
        m_vec = jnp.array( \
            [jnp.sqrt(3.) * x, \
             jnp.sqrt(3.) * y, \
             jnp.sqrt(3.) * z, \
             jnp.sqrt(35./8.)  * (3.*x**2-y**2)*y, \
             jnp.sqrt(105.)    * (x*y*z), \
             jnp.sqrt(21./8.)  * (5.*z**2-1.)*y, \
             jnp.sqrt(7./4.)   * (5.*z**2-3.)*z, \
             jnp.sqrt(21./8.)  * (5.*z**2-1.)*x, \
             jnp.sqrt(105./4.) * (x**2-y**2)*z, \
             jnp.sqrt(35./8.)  * (x**2-3.*y**2)*x, \
             jnp.sqrt(11./64.)    * (63.*z**4-70.*z**2+15.)*z, \
             jnp.sqrt(165./64.)   * (21.*z**4-14.*z**2+1.)*x, \
             jnp.sqrt(165./64.)   * (21.*z**4-14.*z**2+1.)*y, \
             jnp.sqrt(1155./16.)  * (3.*z**2-1.)*(x**2-y**2)*z, \
             jnp.sqrt(1155./4.)   * (3.*z**2-1.)*x*y*z, \
             jnp.sqrt(385./128.)  * (9.*z**2-1.)*(x**2-3.*y**2)*x, \
             jnp.sqrt(385./128.)  * (9.*z**2-1.)*(3.*x**2-y**2)*y, \
             jnp.sqrt(3465./64.)  * (x**4-6.*x**2*y**2+y**4)*z, \
             jnp.sqrt(3465./4.)   * (x**2-y**2)*x*y*z, \
             jnp.sqrt(693./128.)  * (x**4-10.*x**2*y**2+5.*y**4)*x, \
             jnp.sqrt(693./128.)  * (5.*x**4-10.*x**2*y**2+y**4)*y ]) / jnp.sqrt(4.*jnp.pi)
        return sigmoid(jnp.dot(m_vec, hidden_state))
    return rule, hidden_dim

# ---------- ---------- ---------- ----------
# Odd Spherical harmonics up to degree D

def Cl0(l):
    x, y, z = sym.symbols('x y z')
    a = sum([sym.binomial(l, k) * sym.binomial (sym.Rational((l+k-1)/2), l) * z**k for k in range(l+1)])
    a *= sym.sqrt(2*l+1) * 2**l
    return sym.simplify(a)

def CSlm(l, m, bC, bS):
    x, y, z = sym.symbols('x y z')
    a = sum([sym.binomial(l, k+m) * sym.binomial( sym.Rational((l+k+m-1)/2), l) * sym.Rational(sym.factorial(k+m) / sym.factorial(k)) * z**k for k in range(l-m+1)])
    a *= sym.sqrt(2*l+1) * sym.sqrt(2) * 2**l * sym.sqrt( sym.Rational( sym.factorial(l-m) / sym.factorial(l+m)) )
    return sym.simplify(a*bC), sym.simplify(a*bS)

def bCm(m):
    x, y, z = sym.symbols('x y z')
    return sum([(-1)**j * sym.binomial(m, 2*j)   * x**(m-2*j)   * y**(2*j)   for j in range(int( m   /2)+1)])

def bSm(m):
    x, y, z = sym.symbols('x y z')
    return sum([(-1)**j * sym.binomial(m, 2*j+1) * x**(m-2*j-1) * y**(2*j+1) for j in range(int((m-1)/2)+1)])

def spherical_harmonics_expr(D):
    x, y, z = sym.symbols('x y z')
    bC = [bCm(m) for m in range(1, D+1)]
    bS = [bSm(m) for m in range(1, D+1)]
    Sph = []
    for l in range(1, D+1, 2):
        Sph.append(Cl0(l))
        for m in range(1, l+1):
            C, S = CSlm(l, m, bC[m-1], bS[m-1])
            Sph.append(C)
            Sph.append(S)
    return Sph

def make_func(expr):
  x, y, z = sym.symbols('x y z')
  f = sym.lambdify([x, y, z], expr, 'jax')
  return f

def PLHV_sh(D):
    hidden_dim = int(jnp.sum(2*jnp.arange(1, D+1, 2)+1).item())
    Sph_expr = spherical_harmonics_expr(D)
    Sph_funcs = [make_func(expr) for expr in Sph_expr]
    def rule(measure, hidden_state):
        x, y, z = measure
        m_vec = jnp.array( [f(x, y, z) for f in Sph_funcs] ) / jnp.sqrt(4.*jnp.pi)
        return sigmoid(jnp.dot(m_vec, hidden_state))
    return rule, hidden_dim

# ---------- ---------- ---------- ----------


# Without normalization:
def PLHV_sh5_old():
    hidden_dim = 3 + 7 + 11
    def rule(measure, hidden_state):
        x, y, z = measure
        m_vec = jnp.array( \
                   [x, y, z, \
                    3.*x**2*y-y**3, x*y*z, 5.*z**2*y-y, (5./3.)*z**3-z, 5.*z**2*x-x, x**2*z-y**2*z, x**3-3.*y**2*x, \
                    z*((21./5.)*z**4-(14./3.)*z**2+1.), (21.*z**4-14.*z**2+1.)*x, (21.*z**4-14.*z**2+1.)*y, \
                    (3.*z**2-1.)*z*(x**2-y**2), (3.*z**2-1.)*z*x*y, (9.*z**2-1.)*(x**3-3.*x*y**2), (9.*z**2-1.)*(3.*x**2*y-y**3), \
                    z*(x**4-6.*x**2*y**2+y**4), z*(x**3*y-x*y**3), x**5-10.*x**3*y**2+5.*x*y**4, 5.*x**4*y-10.*x**2*y**3+y**5 ])
        return sigmoid(jnp.dot(m_vec, hidden_state))
    return rule, hidden_dim

# Odd Spherical harmonics up to degree 5 for planar measurements (z=0)
def PLHV_spherical_harmonics_planar():
    hidden_dim = 6
    def rule(measure, hidden_state):
        x, y = measure
        m_vec = jnp.array( \
            [jnp.sqrt(3.) * x, \
             jnp.sqrt(3.) * y, \
             jnp.sqrt(35./8.)  * (3.*x**2-y**2)*y, \
             jnp.sqrt(35./8.)  * (x**2-3.*y**2)*x, \
             jnp.sqrt(693./128.)  * (x**4-10.*x**2*y**2+5.*y**4)*x, \
             jnp.sqrt(693./128.)  * (5.*x**4-10.*x**2*y**2+y**4)*y ]) / jnp.sqrt(4.*jnp.pi)
        return sigmoid(jnp.dot(m_vec, hidden_state))
    return rule, hidden_dim


# All odd polynomials up to arbitrary degree:

# All combinations of 3 natural numbers summing to "degree"
def powers(degree): 
    combos = jnp.array(jnp.meshgrid(*jnp.outer(jnp.ones(3, dtype=int), jnp.arange(degree+1, dtype=int)), indexing='ij'))
    combos = jnp.moveaxis(combos, (0,), (-1,))
    combos = jnp.reshape(combos, ((degree+1)**3, 3))
    indices = jnp.where(jnp.sum(combos, axis=-1)==degree)
    return combos[indices]

# Monomial of three variables (vec) with three powers
def monomial(vec, powers):
    return jnp.prod(vec**powers)
monomials = vmap(monomial, in_axes=(None, 0))

# Odd Polynomial measurement rule
def PLHV_polynomial(degree):
    pow = jnp.concatenate([powers(n) for n in range(1, degree+1, 2)], axis=0)
    hidden_dim = pow.shape[0]
    def rule(measure, hidden_state):
        return sigmoid(jnp.dot(hidden_state, monomials(measure, pow)))
    return rule, hidden_dim


############
# Sampling #
############

# key: For random number generator
# N_measures: Batch size ( = number of measurements (per spin))
# N_spins: Number of spins
# returns: Batch of measurement directions for all spins

# All projective measurements, sample uniformely from surface of sphere
def sample3Dprojective(key, N_measures, N_spins):
    batched_measures = random.normal(key, (N_measures, N_spins, 3))
    return batched_measures / jnp.linalg.norm(batched_measures, axis=-1, keepdims=True)

# Projective measurements in xy-plane (works with PQM_Werner like this)
def sample2Dprojective(key, N_measures, N_spins):
    batched_measures_xy = random.normal(key, (N_measures, N_spins, 2))
    return batched_measures_xy / jnp.linalg.norm(batched_measures_xy, axis=-1, keepdims=True)


##################
# Initialization #
##################

# Initialize the hidden states as
# hidden_states_init =random.normal(key, (N_hidden, N_spins, hidden_dim))
# N_hidden: Size of hidden state cloud
# N_spins: Number of spins
# hidden_dim: Dimension of the hidden state vectors (per spin)

