import os

import click
import pandas as pd
import numpy as np
import yaml

config_path = os.environ["CONFIG_PATH_PARAMS"]


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def preprocess(
    input_path: str,
    output_path: str
) -> None:
    # Depending on directory might be "train" or "test"
    task = input_path.split("/")[-1]
    config = yaml.safe_load(open(config_path))["preprocess"]
    DROP_COLS = config["drop_columns"]

    try:
        data_file = pd.read_csv(input_path, parse_dates=["time"])
    except IsADirectoryError:
        print("File is empty, nothing to load")
        return
    data_file = data_file.set_index("time")
    # data_file = data_file.drop(DROP_COLS, axis=1)
    data_file = data_file.replace(float("inf"), np.nan)
    cols = data_file.columns
    row_number = data_file.shape[0]

    # I use train data only to decide which columns to use in next steps
    if task == "train.csv":
        # Drop columns that are mostly NaNs
        for col in cols:
            count_nan = data_file[col].isna().sum()
            if count_nan >= row_number // 5:
                DROP_COLS.append(col)
    DROP_COLS = np.unique(DROP_COLS).tolist()
    data_file = data_file.select_dtypes(include=[int, float])
    data_file = data_file.drop(DROP_COLS, axis=1)
    data_file = data_file.dropna()

    data_file.to_csv(output_path)

    if task == "train.csv":
        # Register necessary columns for resulting dataset
        config = yaml.safe_load(open(config_path))
        config["preprocess"]["drop_columns"] = DROP_COLS

        with open(config_path, "w") as f:
            yaml.dump(config, f, encoding="UTF-8", allow_unicode=True, default_flow_style=False)


if __name__ == "__main__":
    preprocess()
