import os
import runpy
import sys
from pathlib import Path

import click
from pprint import pp


@click.group(
    add_help_option=False,
    invoke_without_command=True,
)
@click.pass_context
def cli(ctx):
    "MALT - Margin Aware Labeling Tool"
    if ctx.invoked_subcommand is None:
        info = ctx.to_info_dict()
        help_text = """"""
        help_text += "  " + info["command"]["help"] + "\n"
        help_text += "\nCommands:\n"
        for command_name, command_info in info["command"]["commands"].items():
            help_text += "  " + command_name + "\t" + command_info["help"] + "\n"
        help_text += "\n"
        help_text += (
            "For help on the subcommands call with `malt <subcommand> --help`\n"
        )
        help_text += "----------\n"
        help_text += "Help message of the MALT tool:\n"
        print(help_text)
        if len(sys.argv) == 1:
            sys.argv.append("--help")
        os.environ["MALT_RELOAD"] = "0"
        runpy.run_module("malt", run_name="__main__", alter_sys=True)


@cli.command(
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run():
    """Run MALT tool (also invoked when no subcommand is present)."""
    del sys.argv[1]
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    os.environ["MALT_RELOAD"] = "0"
    runpy.run_module("malt", run_name="__main__", alter_sys=True)


@cli.group()
def tool():
    """Utility commands."""


@tool.command()
@click.argument("source", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "-d",
    "--directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="output dir (default:cwd)",
)
@click.option(
    "-s",
    "--size",
    type=click.FloatRange(min_open=0.0, max=1.0),
    nargs=2,
    default=(0.2, 0.2),
    help="patch size (w%,h%).",
)
@click.option(
    "-o",
    "--overlap",
    type=click.FloatRange(min_open=0.0, max=1.0),
    nargs=2,
    default=(0.1, 0.1),
    help="patch overlap (%x,%y)",
)
@click.option(
    "--pad/--no-pad",
    is_flag=True,
    default=True,
    help="Pad the image to ensure equal patch size. "
    "Right and bottom will be padded with black, transparent pixels. (default: pad)",
)
def img_split(source, directory, size, overlap, pad):
    """Split an image into overlapping patches."""
    from scripts.split_image import split_image

    num_patch = split_image(source, directory, size, overlap, pad)
    print(f"Created {num_patch} patches in {directory} from {source}")
    return 0


@tool.command()
@click.argument("source", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "-d",
    "--directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="output dir (default:cwd)",
)
@click.option("-s", "--start-index", type=click.INT, help="skip the first N series")
def convert_aeon_ts(source, directory, start_index):
    """Convert an aeon .ts file to a collection of CSV files."""
    from scripts.ts_convert import convert_ts_file

    convert_ts_file(
        input_path=source,
        output_directory=directory,
        start_index=start_index,
    )


@tool.command()
@click.argument("source", type=click.Path(dir_okay=False, path_type=Path))
@click.option("-d", "--dest", type=click.Path(path_type=Path), default=Path.cwd())
@click.option(
    "-p", "--pattern", type=str, default=r"_(\d+)_(\d+)(?:_o(\d+)x(\d+))?\.[^.]+$"
)
@click.option(
    "-i", "--img-folder", type=click.Path(file_okay=False, path_type=Path), default=None
)
@click.option(
    "-r",
    "--radius",
    type=float,
    default=0.2,
    help="[x>0]; if x>1 x=|pixels|; if 0<x<1 x=%patch-width",
)
@click.option(
    "-g",
    "--radius-gauss",
    type=float,
    default=0.5,
    help="[x>0]; if x>1 x=|pixels|; if 0<x<1 x=%patch-width",
)
@click.option(
    "-ps", "--patch-size", type=click.IntRange(min=1), nargs=2, default=(25, 25)
)
@click.option("--posbyidx", is_flag=True, help="patch position is given by index")
def soft_labeling(source, dest, pattern, img_folder, radius, radius_gauss, patch_size, posbyidx):
    """Make a soft labeling from a malt state file or a pandas readable table."""
    from components.utils import load_table, has_header
    from scripts.make_soft_labeling import soft_label

    if source.suffix == ".malt":
        import pandas as pd

        # df = load_labels(source)
        df = pd.DataFrame(columns=["fpath", "cls"])
    else:
        df = load_table(source, header=has_header(source))
    soft_label(df, dest, pattern, img_folder, radius, radius_gauss, patch_size, patch_pos_by_idx=posbyidx)


def main():
    # Only hand control to Click when the first argument names
    # one of its registered top-level commands.
    if len(sys.argv) == 1:
        cli.main(prog_name="malt")
        return
    if len(sys.argv) > 1 and sys.argv[1] in cli.commands:
        cli()
    else:
        os.environ["MALT_RELOAD"] = "0"
        runpy.run_module("malt", run_name="__main__", alter_sys=True)
