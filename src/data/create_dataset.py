import os

import click
import pandas as pd
import yaml

config_path = os.environ["CONFIG_PATH_PARAMS"]
FILENAMES = {"train": "train.csv", "test": "test.csv"}


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def join_data(
    input_path: str,
    output_path: str,
) -> None:
    """
    This function joins all datasets in 'list_data' into one pd.DataFrame and splits it in train and test
    """
    # We use columns for joined dataset from output of expand target script
    config = yaml.safe_load(open(config_path))["create_dataset"]
    data_cols = config["columns"]
    joined_df = pd.DataFrame(columns=data_cols)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path)
        joined_df = pd.concat([joined_df, data_file], axis=0)

    joined_df = joined_df.reset_index(drop=True)
    try:
        joined_df.to_csv(output_path, index=False)
    except IsADirectoryError:
        print("File is empty, nothing to save")


if __name__ == "__main__":
    join_data()
