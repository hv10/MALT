import subprocess

import numpy as np
import re

def embed_file(filepath, state=None):
    """
    Embeds a file by calling the setup external command.
    Assumes the output of the external command can be captured via stdout 
    and is castable as a numpy.array of type np.float32.
    It also assumes a minimum size (D=2) for embeddings.

    !IMPORTANT!: You do have to ensure that the external command 
                 _always_ returns embeddings of the same fixed size.
                 This 1d-size needs to be known _a-priori_ and set in the config.
    !IMPORTANT!: Your command has to be available in the PATH.
    !IMPORTANT!: Your command has to be set accordingly in the build.toml. [embeddings]>"cmd"
    !IMPORTANT!: Your return size has to be set accordingly in the build.toml. [embeddings]>"size"
    """
    try:
        cmd = state.META["cfg"]["embeddings"]["cmd"]
        D = state.META["cfg"]["embeddings"]["size"]
    except KeyError as e:
        print("Error: You have either not set your embedding command `cmd` or the embedding size `size`.")
        raise e
    if isinstance(cmd, str):
        cmd = cmd.replace("{{f}}", str(filepath))
    elif isinstance(cmd, list):
        for i,el in enumerate(cmd):
            if "{{f}}" in el:
                cmd[i] = el.replace("{{f}}", str(filepath))
    else:
        return np.zeros(D, dtype=np.float32)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        shell=True if isinstance(cmd,str) else False
    )

    if result.returncode != 0:
        print(f"Applying {cmd} failed with err_code {result.returncode}.")
        return np.zeros(D,dtype=np.float32)
    
    # cast the output into a numpy array
    # if the output contains ',' '\t' or '\s' we assume that is the separator for entries
    # if the output contains '\n' _but not_ one of the separators, we assume '\n' is the separator
    # if the output contains _both_ '\n' and a separator, we still assume a 1d array we load accordingly
    # Note: if the output contains _exactly_ 2 lines, and the second one is empty, we ignore the '\n' rules
    #       this is so we can support tools that adhere to the standard of writing an empty-line at the 
    #       end of files.
    sep_match = re.search(r"([,\t ])", result.stdout)
    lines = result.stdout.strip().splitlines()
    if sep_match is None:
        print(f"Applying {cmd} did not yield properly delimited content (not delimited by comma, tab, or space)")
        return np.zeros(D,dtype=np.float32)

    sep = sep_match.group(1) if sep_match is not None else None
    
    array = np.fromstring(sep.join(lines), sep=sep, dtype=np.float32)
    return array



if __name__ == "__main__":
    from pathlib import Path
    import tomllib
    import rootpath
    rootpath.append()
    from components.state import State
    state = State()
    state.META |= {"cfg": {"embeddings":{"cmd":["cat", "{{f}}"], "size":128}}}
    emb = embed_file(Path(__file__).parent / "test.txt", state)
    print(emb.shape)
    print(emb)