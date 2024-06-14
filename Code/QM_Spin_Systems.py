# Packages
from numpy import zeros
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import complex
from functools import reduce

# Constants
pi = jnp.pi
i = complex(0., 1.)

# Pauli matrices
up = jnp.array([1., 0.])
down = jnp.array([0., 1.])
identity = jnp.outer(up, up) + jnp.outer(down, down)
sigmaX = jnp.outer(up, down) + jnp.outer(down, up)
sigmaY = i*jnp.outer(down, up) - i*jnp.outer(up, down)
sigmaY_real = jnp.outer(down, up) - jnp.outer(up, down)
sigmaZ = jnp.outer(up, up) - jnp.outer(down, down)
sigma_vec = jnp.array([identity, sigmaX, sigmaY, sigmaZ])
XX = jnp.kron(sigmaX, sigmaX)
YY = -jnp.kron(sigmaY_real, sigmaY_real)
ZZ = jnp.kron(sigmaZ, sigmaZ)
XY = jnp.kron(sigmaX, sigmaY)
XZ = jnp.kron(sigmaX, sigmaZ)
YX = jnp.kron(sigmaY, sigmaX)
YZ = jnp.kron(sigmaY, sigmaZ)
ZX = jnp.kron(sigmaZ, sigmaX)
ZY = jnp.kron(sigmaZ, sigmaY)
one_points = jnp.array([sigmaX, sigmaY, sigmaZ])
two_points = jnp.array([XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ])


#############
# Functions #
#############

def partial_trace(matrix, subsystem, N): # Matrix has to act on N spin-1/2, i.e., an 2^N by 2^N matrix
    # Subsystem is the list of spins we are reducing to, that is the partial trace is over all other spins
    subsystems_to_trace = list(set(range(N))-set(subsystem))
    l = len(subsystems_to_trace)
    M = jnp.reshape(matrix, jnp.repeat(2, 2*N)) # Reshape matrix into (2, 2, 2, ..., 2) tensor, each index corresponds to one spin
    M = jnp.moveaxis(M, subsystems_to_trace+[j+N for j in subsystems_to_trace], list(range(N-l, N))+list(range(2*N-l, 2*N))) # move axis to trace to the end
    M = jnp.reshape(M, (2**(N-l), 2**l, 2**(N-l), 2**l)) # collect axes to trace into matrix (2 indices)
    return jnp.trace(M, axis1=1, axis2=3) # trace over collected axes

def tensorproduct(matrix_list): # Tensor product of a list of matrices
    return reduce(jnp.kron, matrix_list)

# Embed lokal operator A into N spin Hilbert space A -> 1x1x...xAx...x1 (A at position idx...idx+length-1)
def embed(A, idx, length, N): 
    id_left = jnp.diag(jnp.ones(2**idx))
    id_right = jnp.diag(jnp.ones(2**(N-idx-length)))
    return tensorproduct([id_left, A, id_right])
embed_list = vmap(embed, in_axes=(0, None, None, None))

# Correlators for l spins
def correlators0(rho, indices): # there must be l indices if rho is an l qubit state
    pauli_string = tensorproduct(sigma_vec[indices])
    return jnp.trace(rho@pauli_string)
correlators1 = jnp.vectorize(correlators0, excluded=[0], signature="(m)->()")
def correlators(rho, l): # All correlators, ie expectation values of pauli strings, in the state rho, as rank l tensor
    indices = jnp.array(jnp.meshgrid(*jnp.outer(jnp.repeat(1, l),jnp.arange(4)), indexing='ij'))
    indices = jnp.moveaxis(indices, 0, -1)
    return jnp.real(correlators1(rho, indices))
correlators = jit(correlators, static_argnums=1)

# Equivalent to partial trace on the level of the correlations
def correlations_subsystem(corrs, subsystem, N):
    indices = zeros(N, dtype=type(None))
    indices[subsystem] = slice(0, 4)
    return corrs[*indices]

# Construct correlations for neighboring spin subsystems directly from pure state
@jit
def projector_jit(state):
    return jnp.outer(jnp.conjugate(state), state)
def efficient_correlators(state, N, N_total): # For large N_total: sequential due to memory limits
    state = jnp.reshape(state, (2**(N_total-N), 2**N))
    rhos = [projector_jit(s) for s in state]
    corrs = jnp.array([correlators(rho, N) for rho in rhos])
    return jnp.sum(corrs, axis=0)

def projector(state):
    return jnp.outer(jnp.conjugate(state), state)
def efficient_correlators_jit(state, N, N_total): # For N_total<=20, N=3 or N_total<=10, N=6 can use vmap
    state = jnp.reshape(state, (2**(N_total-N), 2**N))
    rhos = vmap(projector)(state)
    corrs = vmap(correlators, in_axes=(0, None))(rhos, N)
    return jnp.sum(corrs, axis=0)
efficient_correlators_jit = jit(efficient_correlators_jit, static_argnums=(1,2))

# Calculate gibbs state densitiy matrix from list of energy eigenstates and corresponding energy eigenvalues
def vectorized_projector(vector):
    return jnp.outer(jnp.conjugate(vector), vector)
vectorized_projector = vmap(vectorized_projector, in_axes=1)
@jit
def gibbs_state(eigVecs, eigVals, beta):
    rho = jnp.dot(jnp.moveaxis(vectorized_projector(eigVecs), 0, -1), jnp.exp(-beta*eigVals))
    return rho/jnp.sum(jnp.exp(-beta*eigVals))

@jit # Hidden state distribution for a single spin in the state rho = rho(r) for some bloch vector r
def p_hidden(bloch_vector, hidden_state):
    return jnp.dot(bloch_vector, hidden_state)\
            *jnp.heaviside(jnp.dot(bloch_vector, hidden_state), 0.5)/jnp.pi \
            + (1.-jnp.linalg.norm(bloch_vector))/(4.*jnp.pi)


################
# Hamiltonians #
################

# XXZ Hamiltonian for 1d chain
def hamiltonian_XXZ(Delta, N):
    H = jnp.zeros((2**N, 2**N))
    for j in range(N-1):
        H +=        embed(XX, j, 2, N) \
                  + embed(YY, j, 2, N) \
            + Delta*embed(ZZ, j, 2, N)
    # periodic boundary conditions:
    if N > 2:
        id_center = jnp.diag(jnp.ones(2**(N-2)))
        H +=        tensorproduct([sigmaX, id_center, sigmaX]) \
                  - tensorproduct([sigmaY_real, id_center, sigmaY_real]) \
            + Delta*tensorproduct([sigmaZ, id_center, sigmaZ])
    return H
hamiltonian_XXZ = jit(hamiltonian_XXZ, static_argnums=1)

# XYZ Hamiltonian (general Heisenberg model) with magnetic fields for 1d chain
def hamiltonian_XYZ_h(J, h, N):
    H = jnp.zeros((2**N, 2**N))
    for j in range(N-1):
        H +=  J[0]*embed(XX, j, 2, N) \
            + J[1]*embed(YY, j, 2, N) \
            + J[2]*embed(ZZ, j, 2, N) \
            + h[0]*embed(sigmaX, j, 1, N) \
            + h[1]*embed(sigmaY, j, 1, N) \
            + h[2]*embed(sigmaZ, j, 1, N)
    # periodic boundary conditions:
    if N > 2:
        id_center = jnp.diag(jnp.ones(2**(N-2)))
        H +=  J[0]*tensorproduct([sigmaX, id_center, sigmaX]) \
            - J[1]*tensorproduct([sigmaY_real, id_center, sigmaY_real]) \
            + J[2]*tensorproduct([sigmaZ, id_center, sigmaZ])
    H +=  h[0]*embed(sigmaX, N-1, 1, N) \
        + h[1]*embed(sigmaY, N-1, 1, N) \
        + h[2]*embed(sigmaZ, N-1, 1, N)
    return H
hamiltonian_XYZ_h = jit(hamiltonian_XYZ_h, static_argnums=2)

# Arbitrary nearest neighbor N qubit chain with periodic boundary conditions
def hamiltonian_1d_nearestNeighbor(J, h, N): # J: spin-spin interactions, h: magnetic fields
    
    H = jnp.zeros((2**N, 2**N))
    
    for j in range(N-1):
        embedded_one_points = embed_list(one_points, j, 1, N)
        embedded_two_points = embed_list(two_points, j, 2, N)
        H +=   jnp.dot(jnp.moveaxis(embedded_two_points, 0, -1), J) \
             + jnp.dot(jnp.moveaxis(embedded_one_points, 0, -1), h)
    
    embedded_one_points = embed_list(one_points, N-1, 1, N)
    H += jnp.dot(jnp.moveaxis(embedded_one_points, 0, -1), h)
    
    if N>2:
        id_center = jnp.diag(jnp.ones(2**(N-2)))
        embedded_two_points = jnp.array([tensorproduct([one_points[j], id_center, one_points[k]]) for k in range(3) for j in range(3)])
        H +=   jnp.dot(jnp.moveaxis(embedded_two_points, 0, -1), J)
    
    return H
hamiltonian_1d_nearestNeighbor = jit(hamiltonian_1d_nearestNeighbor, static_argnums=2)
