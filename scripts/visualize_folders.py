import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def visualize(glob_path: str):
    """
    List the items in the folder that match the given glob path.

    Parameters:
        glob_path (str): The glob path to search for files and folders.

    Example:
        >>> visualize('/path/to/files/*.txt')
        ['file1.txt', 'file2.txt']
    """
    items = glob.glob(glob_path)
    if not items:
        print(f"No items found for the path: {glob_path}")
        return

    rows = {}
    for dir_ in items:

        dir_ = Path(glob_path).joinpath(dir_)


        if dir_.is_dir():

            items = list(dir_.glob("*"))

            # get number identifiers
            nums = []
            for item in items:
                num = item.name.replace(f"{dir_.name}-", "")#[0]

                if re.match(r'^\d+', num):

                    for sep in ["-", ".", "_"]:
                        num = num.split(sep)[0]

                    nums.append(num)

            nums = set(nums)

            for item in items:
                for num in nums:

                    base_name = f"{dir_.name}-{num}"
                    if base_name in item.name:

                        if base_name not in rows.keys():
                            rows[base_name] = []

                        suffix = item.name.replace(base_name, "")
                        if suffix == "":
                            suffix = "dir"

                        rows[base_name].append(suffix)

    # Create a defaultdict with default value 0
    output_dict = defaultdict(lambda: defaultdict(int))

    # Loop through each dictionary in the list
    for key, suffixes in rows.items():
        for suffix in suffixes:
            output_dict[key][suffix] = 1

    # Convert the defaultdict to a pandas DataFrame
    df = pd.DataFrame.from_dict(output_dict, orient='index').fillna(0).astype(int)

    def sort_func(x):

        order = []
        for xx in x:

            number = ""
            for v in xx:
                try:
                    int(v)
                    number += v
                except ValueError:
                    pass

            order.append(int(number))

        return order

    df.sort_index(key=lambda x: sort_func(x), inplace=True)

    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize items in a folder matching a glob pattern.")
    parser.add_argument("glob_path", type=str, help="The glob path to search for files and folders.")

    args = parser.parse_args()
    visualize(args.glob_path)
