import os

import click
import pandas as pd
import yaml

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/params_all.yaml")
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
    # Train or test, we can confirm that by folder name
    folder_to_save = input_path.split("/")[-1]
    folder_path = os.path.join(output_path, folder_to_save)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    config = yaml.safe_load(open(CONFIG_PATH))["create_dataset"]
    data_cols = config["columns"]
    joined_df = pd.DataFrame(columns=data_cols)

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path)
        joined_df = pd.concat([joined_df, data_file], axis=0)

    joined_df = joined_df.reset_index(drop=True)
    joined_df["time"] = pd.to_datetime(joined_df["time"])
    joined_df = joined_df.sort_values(by="time", ascending=True)

    new_name = FILENAMES[folder_path]
    save_path = os.path.join(folder_path, new_name)
    joined_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    join_data()
