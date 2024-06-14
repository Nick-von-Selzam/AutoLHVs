# Functions for visualization

# Packages
import jax.numpy as jnp
from jax import jit, vmap

@jit # Given cloud of vectors (vecs) and a normal vector (n) calculate the sum of gaussians for n-vecs[j] with std delta 
def gaussian_blurr(vecs, n, delta):
    vecs /= jnp.linalg.norm(vecs, axis=-1, keepdims=True)
    return jnp.sum(jnp.exp(-jnp.linalg.norm(vecs - n, axis=-1)**2/(2.*delta)))
gaussian_blurr = vmap(vmap(gaussian_blurr, in_axes=(None, 0, None)), in_axes=(None, 0, None))

@jit # Cartesian coordinates from spherical angles (convention to match matplotlibs mollweide projection)
def normal0(phi, theta): 
    x = jnp.cos(theta)*jnp.cos(phi)
    y = jnp.cos(theta)*jnp.sin(phi)
    z = jnp.sin(theta)
    return jnp.array([x, y, z])
normal = vmap(vmap(normal0, in_axes=(0, 0)), in_axes=(0, 0))

@jit # Spherical angles for normal vector n (convention to match matplotlibs mollweide projection)
def S2_angles0(n): 
    return (jnp.arctan2(n[1], n[0]), jnp.arcsin(n[2]))
S2_angles = vmap(S2_angles0)
