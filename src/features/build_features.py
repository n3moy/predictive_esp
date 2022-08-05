import os
import time
from datetime import datetime

import click
import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureCalculation


# This func should be used after joining data by well
@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("events_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("update_file", type=click.Path())
@click.argument("target_window", type=click.INT)
@click.argument("verbose", type=click.BOOL)
def feature_calculation(
        input_path: str,
        events_path: str,
        output_path: str,
        update_file: str,
        target_window: int = 7,
        verbose: bool = False
) -> None:
    data_lst = []
    events_dict = {}
    common_cols = []

    for cnt, filename in enumerate(os.listdir(input_path)):
        file_path = os.path.join(input_path, filename)
        if verbose:
            print(f"Collecting data {file_path}...")
        csv_file = pd.read_csv(file_path, parse_dates=["time"]).sort_values(by="time")
        data_lst.append(csv_file)
        cols = csv_file.columns
        if cnt == 0:
            common_cols = cols
        common_cols = np.intersect1d(common_cols, cols)

    for f in os.listdir(events_path):
        file_path = os.path.join(events_path, f)
        events = pd.read_csv(file_path, parse_dates=["startDate", "endDate"])
        events_dict[int(f[:-4][-1])] = events

    for joined_data in data_lst:
        try:
            well_id = joined_data["well"].values[0]
        except KeyError:
            well_id = joined_data["well_id"].values[0]
        if verbose:
            print(f"Calculating new features for Well #{well_id}...")
        out_data = joined_data[common_cols].copy()
        out_data["time"] = pd.to_datetime(out_data["time"])
        out_data = out_data.set_index("time")
        out_data = FeatureCalculation.calculate_data_features(out_data)
        out_data = FeatureCalculation.join_events_to_data(out_data, events_dict[well_id])
        out_data = FeatureCalculation.expand_target(out_data, target_window)
        save_name = f"joined_featured_{well_id}.csv"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path_to_save = os.path.join(output_path, save_name)
        out_data.to_csv(path_to_save, index=False)
        if verbose:
            print(f"Output data is successfully saved to '{path_to_save}' !")

    with open(update_file, "a") as f:
        upd_date = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\nUpdate at {upd_date}")


if __name__ == "__main__":
    feature_calculation()
