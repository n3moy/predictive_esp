import os

import click
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("target_window", type=click.INT)
def expand_target(
    input_path: str,
    output_path: str,
    target_window: int
) -> None:

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path, parse_dates=["time"])

        if is_datetime(data_file.index):
            data_file = data_file.reset_index()

        if not is_datetime(data_file["time"]):
            data_file["time"] = pd.to_datetime(data_file["time"])

        data_file.loc[(data_file["event_id"] != 0), "failure_date"] = data_file.loc[
            (data_file["event_id"] != 0), "time"
        ]
        data_file["failure_date"] = data_file["failure_date"].bfill().fillna(data_file["time"].max())
        data_file["failure_date"] = pd.to_datetime(data_file["failure_date"])
        data_file["fail_range"] = data_file["failure_date"] - data_file["time"]
        data_file["time_to_failure"] = data_file["fail_range"] / np.timedelta64(1, "D")
        data_file.loc[data_file["failure_date"] == data_file["time"].max(), "time_to_failure"] = 999
        # I use window between 7 and 0 days before failure
        data_file["failure_target"] = np.where(
            ((data_file["time_to_failure"] <= target_window) & (data_file["time_to_failure"] > 0)),
            1,
            0,
        )
        data_file = data_file.drop(["failure_date", "fail_range"], axis=1)
        # data_file["stable"] = np.where(((data_file["time_to_failure"] >= 30)
        # & (data_file["time_to_failure"] <= 90)), 1, 0)
        data_file = data_file[data_file["time_to_failure"] != 999]
        data_file["stable"] = np.where((data_file["time_to_failure"] >= 20), 1, 0)

        save_path = os.path.join(output_path, filename)
        data_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    expand_target()

