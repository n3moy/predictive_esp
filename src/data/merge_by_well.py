import os

import yaml
import click
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("begin_time", type=click.STRING)
@click.argument("end_time", type=click.STRING)
def merge(
    input_path: str,
    output_path: str,
    begin_time: str,
    end_time: str
) -> None:
    begin_time = pd.to_datetime(begin_time, infer_datetime_format=True)
    end_time = pd.to_datetime(end_time, infer_datetime_format=True)
    common_idx = pd.date_range(begin_time, end_time, freq="2min")

    for dirname, _, filenames in os.walk(input_path):
        joined_data = pd.DataFrame({"time": common_idx})

        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            data_file = pd.read_csv(file_path, parse_dates=["time"])
            cols = data_file.columns

            if "time" not in cols:
                continue

            joined_data = pd.merge(left=joined_data, right=data_file, on="time", how="inner")

        if filenames:
            # Last folder name = well id (TMP)
            well_id = dirname.split("/")[-1]
            # folder_path = os.path.join(folder_path, well_id)
            save_filename = f"merged_{well_id}.csv"
            save_path = os.path.join(output_path, save_filename)
            joined_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    merge()
