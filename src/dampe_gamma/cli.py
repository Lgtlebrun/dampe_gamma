"""DAMPE gamma ray pipeline: ``python -m dampe_gamma <command> ...``.

Each command lives in its own module exposing ``add_arguments(parser)`` and
``run(args, paths)``. Modules are imported only when their command runs.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from dampe_gamma.config import Paths

COMMANDS: dict[str, tuple[str, str]] = {
    "split": (
        "selection.split_and_weight",
        "split the MC per cos(theta) bin and add spectral weights",
    ),
    "filter": ("selection.filter", "apply the photon selection to one cos(theta) bin"),
    "cutflow": ("selection.cutflow", "plot the spectrum after each selection cut"),
    "effective-area": ("irf.effective_area", "tabulate the effective area"),
    "psf": ("irf.psf", "fit the point spread function"),
    "energy-dispersion": ("irf.energy_dispersion", "fit the energy dispersion"),
    "observing-time": (
        "exposure.observing_time",
        "observing time of a sky position in detector coordinates",
    ),
    "exposure-map": ("exposure.exposure_map", "daily exposure maps, all sky or around a target"),
    "sum-exposure": ("exposure.sum_maps", "sum daily exposure maps"),
    "counts-map": ("flux.counts_map", "all sky counts map of the flight data"),
}


def _load(command: str):
    return importlib.import_module(f"dampe_gamma.{COMMANDS[command][0]}")


def build_parser(argv: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dampe_gamma", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    requested = next((a for a in argv if a in COMMANDS), None)
    for name, (_, help_text) in COMMANDS.items():
        command_parser = sub.add_parser(name, help=help_text, description=help_text)
        if name == requested:
            _load(name).add_arguments(command_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser(argv).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load(args.command).run(args, Paths.from_env())
