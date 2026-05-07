# MALT - Ultra Fast Labelling Tool
> I have a bunch of images I need to label - fast. 
> I do not want to train / setup any model for it right now.

<img width="1562" height="1015" alt="malt_header" src="https://github.com/user-attachments/assets/a69d73de-4175-487f-9742-27999253c6ff" />


To run the tool use the package manager `uv` and the command

```bash
uv run malt.py
```

with the appropriate options (`uv run malt.py --help`):

```
Reading inline script metadata from `malt.py`
usage: malt.py [-h] [--model MODEL] [-d DIRECTORY] [--color-blind] [--prior] [--prepare] [-f] [--theme {dark,light}]

options:
  -h, --help            # show this help message and exit
  --model MODEL         # Which model to use for embedding (huggingface `repo/name`)
  -d DIRECTORY, --directory DIRECTORY
  --color-blind         # use colorblind friendly colors
  --prior               # assume there is already _some_ state for the folder
  --prepare             # prepare the data in the folder for labeling, without starting MALT
  -f, --fullscreen
  --theme {dark,light}
```

Unzip the data in `data/fashion_small_treaser.zip`, then use the `--prior` mode to inspect the labeling states used for figures in the paper.
