import tqdm
import pandas as pd

def apply_emb_to_df(out_dir, df):
    embed_func = getattr(apply_emb_to_df, "embed_func", None)
    if embed_func is None:
        raise ValueError("Embedding function not set. Please set apply_emb_to_df.embed_func before calling this function.")
    
    tqdm.pandas(desc="Embedding Samples")

    def helper(row):
        row["emb"] = embed_func(out_dir / row["fpth"])
        return row

    return df.progress_apply(helper, axis=1)
