import cuvarbase.bls as bls
from .grid import frequency_grid



def bls(t, y, dy, freqs, fast=False, **search_params):
    if fast:
        return bls.eebls_gpu_fast(t, y, dy, freqs,
                                **search_params)
    else:
        return bls.eebls_gpu(t, y, dy, freqs,
                                **search_params)
