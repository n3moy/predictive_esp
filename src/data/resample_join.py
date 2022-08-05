import os
import time
from datetime import datetime

import click
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("update_file", type=click.Path())
@click.argument("verbose", type=click.BOOL)
def join_data_by_well(
        input_path: str, output_path: str, update_file: str, verbose: bool = False
) -> None:

    for dirname, _, filenames in os.walk(input_path):
        if filenames:
            full_data = pd.DataFrame(columns=["time", "value", "feature"])
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            csv_file = pd.read_csv(file_path)
            cols = csv_file.columns

            if "time" not in cols:
                print(f"{filename} doesn't have 'time' column")
                continue
            if not is_datetime(csv_file["time"]):
                csv_file["time"] = pd.to_datetime(csv_file["time"])
            feature_name = cols[1]
            csv_file = csv_file.rename(columns={feature_name: "value"})
            csv_file["feature"] = feature_name
            csv_file = csv_file.sort_values(by="time")
            full_data = full_data.append(csv_file)
        if filenames:
            duplicates_indices = full_data.duplicated(["time", "feature"], keep="first")
            full_data = full_data[~duplicates_indices]
            full_data_pivot = full_data.pivot(
                index="time", columns="feature", values="value"
            )
            full_data_pivot = full_data_pivot.ffill()
            well_id = dirname[-1]
            if verbose:
                print("Resampling...")
                full_data_pivot = full_data_pivot.resample("2Min").interpolate()
            save_name = f"joined_data_{well_id}.csv"
            save_path = os.path.join(output_path, save_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            full_data_pivot["well_id"] = well_id
            full_data_pivot.to_csv(save_path)
            print(f"Well #{well_id} is saved to {save_path}")

    with open(update_file, "a") as f:
        upd_date = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\nUpdate at {upd_date}")


if __name__ == "__main__":
    join_data_by_well()

