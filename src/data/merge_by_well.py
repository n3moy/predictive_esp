import os

import yaml
import click
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

CONFIG_PATH = "/c/py/predictive_esp/config/params_all.yaml"
config_data = yaml.safe_load(open(CONFIG_PATH))["resample_data"]


@click.command()
@click.argument("input_path", type=click.PATH())
@click.argument("output_path", type=click.PATH())
def merge(
    input_path: str,
    output_path: str
) -> None:
    begin_time = pd.to_datetime(["begin_time"])
    end_time = pd.to_datetime(["end_time"])
    common_idx = pd.date_range(begin_time, end_time)

    # Train or test, we can confirm that by parent folder name
    folder_to_save = input_path.split("/")[-1]
    folder_path = os.path.join(output_path, folder_to_save)
    joined_data = pd.DataFrame(index=common_idx)

    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            data_file = pd.read_csv(file_path)
            cols = data_file.columns

            if "time" not in cols:
                continue
            if not is_datetime(data_file["time"]):
                data_file["time"] = pd.to_datetime(data_file["time"])

            joined_data = joined_data.merge(data_file, right_on=joined_data.index, left_on="time")
            joined_data = joined_data.drop("time", axis=1)

        # Last folder name = well id (TMP)
        well_id = dirname.split("/")[-1]
        folder_path = os.path.join(folder_path, well_id)
        save_filename = f"joined_{well_id}.csv"
        save_path = os.path.join(folder_path, save_filename)
        joined_data.to_csv(save_path)


