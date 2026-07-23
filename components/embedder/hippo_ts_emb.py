from pathlib import Path

import numpy as np
import pandas as pd
import rootpath

rootpath.append()

from components.utils import has_header


def ensure_hippo_env():
    import juliapkg

    juliapkg.add(
        "HiPPO",
        "ef35d8a1-c482-4307-aa9a-1639c2824a6d",
        url="https://github.com/hv10/HiPPO.jl",
        rev="0.4.3",
    )

    juliapkg.resolve()


def setup_embed_func(seed=42, N=128, gamma=0.1):
    """Sets up the embedding function using HiPPO.jl and returns it.
    Args:
        seed (int): Random seed for reproducibility.
        N (int): Dimension of the HiPPO state.
        gamma (float): Parameter for the HiPPO transition.
    Returns:
        embed_ts (jl.Callable):
            A julia-function that takes a time-series array and optionally
            the time-steps of the ts-array and returns its embedding.
    """
    import os

    os.environ["PYTHON_JULIACALL_STARTUP_FILE"] = "no"
    os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "3"
    from juliacall import Main as jl

    # for now we magic-number to keep around 10% of the time-series info in the embedding
    # and use a 128d state (compact representation)

    jl.seval("using HiPPO")
    jl.seval("using Random, LinearAlgebra, Statistics")
    jl.seval(f"""
        # Setting up globals
        const N = {N}
        const seed = {seed}
        rng = MersenneTwister(seed)
        const W = randn(rng, Float32, N, N) 
        const b = rand(rng, Float32, N) .* (2f0*Float32(pi))
        const A,B = HiPPO.transition(:legs, N, HiPPO.get_gamma(Float32({gamma})))
    """)
    jl.seval("""
        phi(x) = begin
            col_norms = sqrt.(sum(abs2, x; dims=1))
            x = x ./ max.(col_norms, 1f-6) # normalize to common norm, while controlling for numerical stability
            res = Float32[]
            for sigma in [sqrt(0.1f0), sqrt(0.5f0), 1.0f0]
                x = cos.((W ./ sigma) * x .+ b)
                append!(res, mean(x; dims=2))
            end
            return res
        end
    """)
    embed_ts = jl.seval("""
        function embed_ts(ts_arr; time_steps=Float32.(1:length(ts_arr)) ./ length(ts_arr))
            # Convert the input array to a Julia array
            state = zeros(Float32, N, size(ts_arr, 2))
            ts_deltas = [time_steps[1]; diff(time_steps)]
            for (i, v) in zip(ts_deltas, eachrow(ts_arr))
                for c in axes(state,2)
                    state[:,c] = HiPPO.step(:tustin, A,B, state[:,c], v[c], Float32(i))
                end
            end
            # summary features
            col_norms = sqrt.(sum(abs2, state; dims=1))
            amplitudes = log1p.(col_norms)
            mean_amplitude = tanh(mean(amplitudes) / 5f0)
            col_count = log1p(Float32(size(state, 2)))
            return vcat(phi(state)..., mean_amplitude, col_count)
        end
    """)
    return embed_ts


def load_hippo():
    hippo_func = getattr(load_hippo, "func", None)
    if hippo_func is None:
        ensure_hippo_env()
        load_hippo.func = setup_embed_func()
    return load_hippo.func  # type: ignore


def minmax_scale(arr):
    """
    Scales the input array to the range [0, 1] using min-max scaling.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)  # Avoid division by zero
    return (arr - min_val) / (max_val - min_val)


def embed_ts(filepath, state=None):
    """
    Embeds an time-series as a vector using HIPPO.jl.
    If file contains a single column, it is treated as a single time-series, with equal steps.
    If file contains multiple columns, each column is treated as a separate variate of the time-series.
    Note: In this case the first column is assumed to be the time-axis.
    """
    ts_arr = pd.read_csv(filepath, header=has_header(filepath)).to_numpy()
    if ts_arr.shape[1] == 1:
        embedding = load_hippo()(np.astype(ts_arr, np.float32))
    else:
        time_steps = minmax_scale(ts_arr[:, 0])
        ts_data = np.astype(ts_arr[:, 1:], np.float32)
        embedding = load_hippo()(ts_data, time_steps=time_steps)
    return np.asarray(embedding)
