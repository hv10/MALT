import numpy as np
import pandas as pd
from pathlib import Path
import tomllib


def ensure_hippo_env():
    import juliapkg

    juliapkg.add(
        "HiPPO",
        "ef35d8a1-c482-4307-aa9a-1639c2824a6d",
        url="https://github.com/hv10/HiPPO.jl",
        rev="0.4.3",
    )

    juliapkg.resolve()

def setup_embed_func():
    import os
    os.environ["PYTHON_JULIACALL_STARTUP_FILE"] = "no"
    os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "3"
    from juliacall import Main as jl

    # for now we magic-number this to keep around 10% of the time-series in the embedding
    # and use a 128d state (compact representation)

    jl.seval("using HiPPO")
    embed_ts = jl.seval("""
        function embed_ts(ts_arr; time_steps=Float32.(1:length(ts_arr)) ./ length(ts_arr))
            N = 128
            A,B = HiPPO.transition(:legs, N, HiPPO.get_gamma(0.1f0))
            # Convert the input array to a Julia array
            state = zeros(Float32, N, size(ts_arr, 2))
            ts_deltas = [time_steps[1]; diff(time_steps)]
            for (i, v) in zip(ts_deltas, eachrow(ts_arr))
                for c in axes(state,2)
                    state[:,c] = HiPPO.step(:tustin, A,B, state[:,c], v[c], Float32(i))
                end
            end
            return vec(state)
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

def embed_ts(filepath):
    """
    Embeds an time-series as a vector using HIPPO.jl.
    If file contains a single column, it is treated as a single time-series, with equal steps.
    If file contains multiple columns, each column is treated as a separate time-series.
    The first column is assumed to be the time-axis.
    """
    ts_arr = pd.read_csv(filepath, header=None).to_numpy()
    if ts_arr.shape[1] == 1:
        embedding = load_hippo()(ts_arr)
    else:
        time_steps = minmax_scale(ts_arr[:, 0])
        ts_data = ts_arr[:, 1:]
        embedding = load_hippo()(ts_data, time_steps=time_steps)
    return np.asarray(embedding)


if __name__ == "__main__":
    # Example usage
    ts_file = Path(__file__).parent / "example_timeseries.csv"  # Replace with your time-series file path
    embedding = embed_ts(ts_file)
    print("Embedding shape:", embedding.shape)
    print("Embedding:\n", embedding)
