# MALT - Ultra Fast Labelling Tool
> I have a bunch of images I need to label - fast. 
> I do not want to train / setup any model for it right now.

To run the tool use the package manager `uv` and the command

```bash
uv run malt.py
```

with the appropriate options (`uv run malt.py --help`):

```
Reading inline script metadata from `malt.py`
usage: malt.py [-h] [--model MODEL] [-m MARGIN] [-t {classification,regression}] [-d DIRECTORY] [--color-blind] [--prior] [-f] [--theme {dark,light}]

options:
  -h, --help            show this help message and exit
  --model MODEL         Which model to use for embedding.
  -t {classification,regression}, --task {classification,regression}
  -d DIRECTORY, --directory DIRECTORY
  --color-blind
  --prior
  -f, --fullscreen
  --theme {dark,light}
```

Unzip the data in `data/fashion_small_treaser.zip`, then user the prior mode to inspect the labeling states used for figures in the paper.