import os
import runpy
import sys

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
        help_text += "\t" + info["command"]["help"] + "\n"
        help_text += "\nCommands:\n"
        for command_name, command_info in info["command"]["commands"].items():
            help_text += "\t" + command_name + ": "+ command_info["help"] + "\n"
        help_text += "\n"
        help_text += "For help on the subcommands call with `malt <subcommand> --help`\n"
        help_text += "Help message of the MALT tool:\n"
        print(help_text.expandtabs(2))
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
def img_split() -> None:
    """Split an image."""
    print("Image Split")


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
