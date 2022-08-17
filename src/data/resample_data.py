import os
# import time
# from datetime import datetime

import yaml
import click
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
# @click.argument("update_file", type=click.Path())

CONFIG_PATH = "/c/py/predictive_esp/config/params_all.yaml"
config_data = yaml.safe_load(open(CONFIG_PATH))["resample_data"]


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("verbose", type=click.BOOL)
def resample_data(
    input_path: str,
    output_path: str,
) -> None:
    begin_time = pd.to_datetime(["begin_time"])
    end_time = pd.to_datetime(["end_time"])

    # Train or test, we can confirm that by parent folder name
    folder_to_save = input_path.split("/")[-1]
    folder_path = os.path.join(output_path, folder_to_save)

    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            data_file = pd.read_csv(file_path)
            cols = data_file.columns

            if "time" not in cols:
                continue
            if not is_datetime(data_file["time"]):
                data_file["time"] = pd.to_datetime(data_file["time"])

            new_idx = pd.date_range(begin_time, end_time, freq="2M")
            data_file = data_file.reindex(new_idx, method="nearest", limit=1).interpolate(method="spline", order=2)
            data_file = data_file.reset_index().rename(columns={"index": "time"})
            save_path = os.path.join(output_path, save_name)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            full_data_pivot.to_csv(save_path)

            # Last folder name = well id (TMP)
            well_id = dirname.split("/")[-1]
            folder_path = os.path.join(folder_path, well_id)
            save_filename = f"joined_{well_id}.csv"
            save_path = os.path.join(folder_path, save_filename)
            joined_data.to_csv(save_path)

    # with open(update_file, "a") as f:
    #     upd_date = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    #     f.write(f"\nUpdate at {upd_date}")


if __name__ == "__main__":
    resample_data()

