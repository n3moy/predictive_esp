import os
import time
from datetime import datetime

import click
import pandas as pd


def rename_data_features(
    input_data: pd.DataFrame,
    feature_name: str,
    columns: list
) -> pd.DataFrame:
    """
    Basically converts camelCase names into snake_case

    """
    data_out = input_data.copy()
    new_name = ""

    for i, char in enumerate(feature_name):
        if feature_name == "voltageAC":
            new_name = "voltageCA"  # Mistake in initial data
            break
        if feature_name == "activePower1":
            new_name = "active_power"  # Mistake in initial data
            break
        if "Gage" in feature_name:
            new_name = "electricity_gage"  # Mistake in initial data
            break
        if char.isupper() and not (
                char == feature_name[-2] or char == feature_name[-1]
        ):
            if i == 0:
                new_name += char.lower()
            else:
                new_name += "_" + char.lower()
        else:
            new_name += char

    data_out = data_out.rename(columns={columns[0]: "time", columns[1]: new_name})
    return data_out


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("update_file", type=click.Path())
@click.argument("verbose", type=click.BOOL)
def features_renaming(
    input_path: str,
    output_path: str,
    update_file: str,
    verbose: bool = False
) -> None:
    """
    From input directory imports all the files containing 2 columns,
    renames 'value' as <filename> and 'vUpdateTime' as 'time'

    Saves all renamed files into './data/interim/renamed'

    """
    for dirname, _, filenames in os.walk(input_path):
        if filenames:
            out_path = output_path + "\\" + dirname[-1]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            csv_file = pd.read_csv(file_path)
            cols = csv_file.columns
            if len(cols) != 2:
                if verbose:
                    print(f"Skipping file {file_path}\nFile has more than 2 columns")
                continue

            csv_file = rename_data_features(
                input_data=csv_file,
                feature_name=filename.split(".")[0],
                columns=cols,
            )
            save_path = os.path.join(out_path, filename)
            if verbose:
                print(f"Saving a file {filename} into directory\n{save_path}")
            csv_file.to_csv(save_path, index=False)

    # with open(update_file, "a") as f:
    #     upd_date = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    #     f.write(f"\nUpdate at {upd_date}")


if __name__ == "__main__":
    features_renaming()
