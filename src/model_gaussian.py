from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, vmap

from miko.util.extrapolate import interp_lin_farray

def model_func(
    grid_s,
    grid_obs,
    ell,
    Cl,
    need_alias: bool = False,
    Cl_method: str = "cholesky",
    beam=None
):
    npatch = 1
    ntracer = 1
    # Synthesize kappa maps
    # Draw field values in non-centered parametrization.
    kappaGr_raw = numpyro.sample(f"kappaGr_raw_", dist.Normal(loc=jnp.zeros((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1])), scale=jnp.ones((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1]))))
    kappaGi_raw = numpyro.sample(f"kappaGi_raw_", dist.Normal(loc=jnp.zeros((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1])), scale=jnp.ones((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1]))))

    if Cl_method == "cholesky":
        # Cholesky-decomposition for each ell block and map onto the grid
        kappaGr, kappaGi = Cl_cholesky_synthesize(Cl/grid_s.alpha, grid_s.spl_lindex, kappaGr_raw, kappaGi_raw)
    elif Cl_method == "eigen":
        # Eigen-decomposition for each ell block and map onto the grid
        kappaGr, kappaGi = Cl_eigh_synthesize(Cl/grid_s.alpha, grid_s.spl_lindex, kappaGr_raw, kappaGi_raw)
    else:
        raise ValueError("Cl_method must be one of 'cholesky' or 'eigen'")

    # Reshape the data and transform back to fft.fftfreq ordering, preparing for ifft.
    # By grid construction, ycoord is from fft.fftfreq directly, but xcoord is obtained under fftshift
    kappa_lsp_a = jnp.reshape(kappaGr + 1.0j * kappaGi, (npatch, ntracer) + grid_s.lmap_shape)
    
    # 3. Add grid effects.
    # scaling and grid smoothing
    kappa_lsp_a /= jnp.sqrt(grid_obs.alpha / grid_s.alpha)
    
    if beam is not None:
        kappa_lsp_a = jnp.einsum("xy,ptxy->ptxy", beam, kappa_lsp_a)

    if need_alias:
        # Apply aliasing, this gives the maps on the lowres padded grid in Fourier space
        kappa_lsp = grid_s.alias(m=kappa_lsp_a)
    else:
        # drop the irrelevant high frequency modes
        kappa_lsp = grid_s._drop_alias_extension(m=kappa_lsp_a)
        
    # 4. Convert to real space kappa, gamma maps.
    kappa_lsp = jnp.fft.ifftshift(kappa_lsp, axes=(2,))
    kappa_rsp = numpyro.deterministic(f"kappa_", jnp.fft.irfft2(kappa_lsp))
    
    return

def Cl_cholesky_synthesize(Cl_blk, lindex, kappaGr_raw, kappaGi_raw):
    # Cholesky-decomposition for ell blocks.
    Cl_blk /= 2.0  # real/imag split
    Cl_chol = jnp.take(jnp.linalg.cholesky(Cl_blk), lindex, axis=0)
    # Cl_gauss_chol.shape = (lmap_size_for_the_region, ntracer, ntracer)
    # kappa_raw.shape = (npatch, ntracer, lmap_size_for_the_region)
    # Indices names:
    # Number of patches: p
    # Number of tracers: t, s
    # Number of ell modes: n
    # Number of lmap pixel: l
    kappaGr = jnp.einsum("lts,psl->ptl", Cl_chol, kappaGr_raw)
    kappaGi = jnp.einsum("lts,psl->ptl", Cl_chol, kappaGi_raw)
    return kappaGr, kappaGi

def Cl_eigh_synthesize(Cl_blk, lindex, kappaGr_raw, kappaGi_raw):
    # Eigen-decomposition for ell blocks.
    Cl_blk /= 2  # real/imag split
    L_blk, U_blk = jnp.linalg.eigh(Cl_blk)
    # Correct for any negative values.
    L_blk = jnp.where(L_blk <= 0, 1e-10, L_blk)
    # Compute square root.
    L_blk = jnp.sqrt(L_blk)
    # print(L_blk.shape) # (#ell modes, ntracer)
    # print(U_blk.shape) # (#ell modes, ntracer, ntracer)

    # Mapping the blk matrices (eigen decomposition of covs) onto the grid.
    L = jnp.take(L_blk, lindex, axis=0)
    U = jnp.take(U_blk, lindex, axis=0)
    # L.shape = (lmap_size, ntracer)
    # U.shape = (lmap_size, ntracer, ntracer)

    # U.shape = (lmap_size_for_the_region, ntracer, ntracer)
    # kappa_raw.shape = (npatch, ntracer, lmap_size_for_the_region)
    # kappa.shape = (npatch, ntracer, lmap_size_for_the_region)
    # Indices names:
    # Number of patches: p
    # Number of tracers: t, s
    # Number of ell modes: n
    # Number of lmap pixel: l
    kappaGr = jnp.einsum("lt,ptl->ptl", L, kappaGr_raw)
    kappaGr = jnp.einsum("lts,psl->ptl", U, kappaGr)
    kappaGi = jnp.einsum("lt,ptl->ptl", L, kappaGi_raw)
    kappaGi = jnp.einsum("lts,psl->ptl", U, kappaGi)
    return kappaGr, kappaGi
