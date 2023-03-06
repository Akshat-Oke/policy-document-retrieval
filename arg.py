import argparse

parser = argparse.ArgumentParser(description="Process some files.")

# parser.add_argument("filenames", nargs="+", help="helping strings")
# parser.
# add positional arguments for two file paths
parser.add_argument("filepaths", help="path to first file", nargs="+")
# parser.add_argument("filepath2", help="path to second file")

# add an optional argument for a flag to use a saved file
parser.add_argument(
    "-use-saved", metavar="filepath", help="use a saved file instead of two input files"
)

args = parser.parse_args()

# check if the -use-saved flag is present
if args.use_saved:
    filepath = args.use_saved
    print(f"Using saved file at {filepath}")
    print(args.filepaths)
else:
    filepath1 = args.filepaths
    # print(f"Processing files: {filepath1}, {filepath2}")
    print(args.filepaths)
