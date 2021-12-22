#!/usr/bin/env python

from treed import TreeD,__version__
import argparse

desc = f"""
    TreeD {__version__} - Draw a visual representation of the branch-and-cut tree of SCIP for
    a particular instance using spatial dissimilarities of the node LP solutions.
    """

parser = argparse.ArgumentParser(description=desc, prog="treed")

parser.add_argument("model", type=str, help="path to model")
parser.add_argument(
    "--version",
    "-v",
    action="version",
    version=f"TreeD {__version__}",
    help="show version",
)
parser.add_argument(
    "--classic",
    "-c",
    action="store_true",
    help="draw classical 2D tree ignoring spatial information of node LP solutions",
)
parser.add_argument(
    "--transformation",
    "-t",
    choices=["mds", "tsne", "lle", "spectral", "ltsa"],
    default="mds",
    help="2D transformation algorithm for node LP solutions",
)
parser.add_argument(
    "--hidecuts",
    action="store_true",
    help="hide cutting rounds and only show final node LP solutions",
)

parser.add_argument(
    "--quiet", "-q", action="store_true", help="hide solving and progress output",
)

parser.add_argument(
    "--nodelimit", "-n", type=int, default=500, help="node limit for solving the model"
)
parser.add_argument(
    "--setfile", "-s", default=None, help="path to SCIP settings file",
)

args = parser.parse_args()

if args.version:
    print(f"Treed {__version__}")
    exit()

treed = TreeD(
    probpath=args.model,
    transformation=args.transformation,
    showcuts=~args.hidecuts,
    nodelimit=args.nodelimit,
    verbose=~args.quiet,
    setfile=args.setfile,
)

treed.solve()
if args.classic:
    treed.draw2d()
else:
    treed.draw()
