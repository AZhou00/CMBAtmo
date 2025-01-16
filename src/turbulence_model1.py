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
    Cl_CMB,
    Cl_atmo,
    need_alias: bool = False,
    Cl_method: str = "cholesky",
    beam=None,
    shift=None,
    n_exposure=1
):
    npatch = 1
    ntracer = 1
    
    # Synthesize TCMB and Tatmo maps
    # Draw field values in non-centered parametrization.
    TCMBGr_raw = numpyro.sample(f"TCMBGr_raw_", dist.Normal(loc=jnp.zeros((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1])), scale=jnp.ones((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1]))))
    TCMBGi_raw = numpyro.sample(f"TCMBGi_raw_", dist.Normal(loc=jnp.zeros((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1])), scale=jnp.ones((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1]))))
    TatmoGr_raw = numpyro.sample(f"TatmoGr_raw_", dist.Normal(loc=jnp.zeros((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1])), scale=jnp.ones((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1]))))
    TatmoGi_raw = numpyro.sample(f"TatmoGi_raw_", dist.Normal(loc=jnp.zeros((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1])), scale=jnp.ones((npatch, ntracer, grid_s.lmap_shape[0] * grid_s.lmap_shape[1]))))

    if Cl_method == "cholesky":
        # Cholesky-decomposition for each ell block and map onto the grid
        TCMBGr, TCMBGi = Cl_cholesky_synthesize(Cl_CMB/grid_s.alpha, grid_s.spl_lindex, TCMBGr_raw, TCMBGi_raw)
        TatmoGr, TatmoGi = Cl_cholesky_synthesize(Cl_atmo/grid_s.alpha, grid_s.spl_lindex, TatmoGr_raw, TatmoGi_raw)
    elif Cl_method == "eigen":
        # Eigen-decomposition for each ell block and map onto the grid
        TCMBGr, TCMBGi = Cl_eigh_synthesize(Cl_CMB/grid_s.alpha, grid_s.spl_lindex, TCMBGr_raw, TCMBGi_raw)
        TatmoGr, TatmoGi = Cl_eigh_synthesize(Cl_atmo/grid_s.alpha, grid_s.spl_lindex, TatmoGr_raw, TatmoGi_raw)
    else:
        raise ValueError("Cl_method must be one of 'cholesky' or 'eigen'")

    # Reshape the data and transform back to fft.fftfreq ordering, preparing for ifft.
    # By grid construction, ycoord is from fft.fftfreq directly, but xcoord is obtained under fftshift
    TCMB_lsp_a = jnp.reshape(TCMBGr + 1.0j * TCMBGi, (npatch, ntracer) + grid_s.lmap_shape)
    Tatmo_lsp_a = jnp.reshape(TatmoGr + 1.0j * TatmoGi, (npatch, ntracer) + grid_s.lmap_shape)
    
    # 3. Add grid effects.
    # scaling and grid smoothing
    TCMB_lsp_a /= jnp.sqrt(grid_obs.alpha / grid_s.alpha)
    Tatmo_lsp_a /= jnp.sqrt(grid_obs.alpha / grid_s.alpha)
    
    if n_exposure > 1:
        for i in range(n_exposure):
            assert shift is not None, "shift must be provided if n_exposure > 1"
            
            # shift the atmosphere map
            Tatmo_lsp_a_shifted = Tatmo_lsp_a * shift**i
            
            T_lsp_a = TCMB_lsp_a + Tatmo_lsp_a_shifted
            
            # apply beam
            if beam is not None:
                T_lsp_a = jnp.einsum("xy,ptxy->ptxy", beam, T_lsp_a)
            if need_alias:
                # Apply aliasing, this gives the maps on the lowres padded grid in Fourier space
                T_lsp_a = grid_s.alias(m=T_lsp_a)
            else:
                # drop the irrelevant high frequency modes
                T_lsp = grid_s._drop_alias_extension(m=T_lsp_a)
                
            # 4. Convert to real space TCMB, gamma maps.
            T_lsp = jnp.fft.ifftshift(T_lsp, axes=(2,))
            T_rsp = numpyro.deterministic(f"T_{i}", jnp.fft.irfft2(T_lsp))

    # compute the original map for reference
    if beam is not None:
        TCMB_lsp = jnp.einsum("xy,ptxy->ptxy", beam, TCMB_lsp_a)
        Tatmo_lsp = jnp.einsum("xy,ptxy->ptxy", beam, Tatmo_lsp_a)
    if need_alias:
        TCMB_lsp = grid_s.alias(m=TCMB_lsp)
        Tatmo_lsp = grid_s.alias(m=Tatmo_lsp)
    else:
        TCMB_lsp = grid_s._drop_alias_extension(m=TCMB_lsp)
        Tatmo_lsp = grid_s._drop_alias_extension(m=Tatmo_lsp)
    TCMB_lsp = jnp.fft.ifftshift(TCMB_lsp_a, axes=(2,))
    TCMB_rsp = numpyro.deterministic(f"TCMB", jnp.fft.irfft2(TCMB_lsp))
    Tatmo_lsp = jnp.fft.ifftshift(Tatmo_lsp_a, axes=(2,))
    Tatmo_rsp = numpyro.deterministic(f"Tatmo", jnp.fft.irfft2(Tatmo_lsp))
    return

def Cl_cholesky_synthesize(Cl_blk, lindex, TCMBGr_raw, TCMBGi_raw):
    # Cholesky-decomposition for ell blocks.
    Cl_blk /= 2.0  # real/imag split
    Cl_chol = jnp.take(jnp.linalg.cholesky(Cl_blk), lindex, axis=0)
    # Cl_gauss_chol.shape = (lmap_size_for_the_region, ntracer, ntracer)
    # TCMB_raw.shape = (npatch, ntracer, lmap_size_for_the_region)
    # Indices names:
    # Number of patches: p
    # Number of tracers: t, s
    # Number of ell modes: n
    # Number of lmap pixel: l
    TCMBGr = jnp.einsum("lts,psl->ptl", Cl_chol, TCMBGr_raw)
    TCMBGi = jnp.einsum("lts,psl->ptl", Cl_chol, TCMBGi_raw)
    return TCMBGr, TCMBGi

def Cl_eigh_synthesize(Cl_blk, lindex, TCMBGr_raw, TCMBGi_raw):
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
    # TCMB_raw.shape = (npatch, ntracer, lmap_size_for_the_region)
    # TCMB.shape = (npatch, ntracer, lmap_size_for_the_region)
    # Indices names:
    # Number of patches: p
    # Number of tracers: t, s
    # Number of ell modes: n
    # Number of lmap pixel: l
    TCMBGr = jnp.einsum("lt,ptl->ptl", L, TCMBGr_raw)
    TCMBGr = jnp.einsum("lts,psl->ptl", U, TCMBGr)
    TCMBGi = jnp.einsum("lt,ptl->ptl", L, TCMBGi_raw)
    TCMBGi = jnp.einsum("lts,psl->ptl", U, TCMBGi)
    return TCMBGr, TCMBGi
