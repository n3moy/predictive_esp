import os

import yaml
import click
import pandas as pd

CONFIG_PATH = "/c/py/predictive_esp/config/params_all.yaml"
config_data = yaml.safe_load(open(CONFIG_PATH))["resample_data"]


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def resample_data(
    input_path: str,
    output_path: str,
) -> None:
    begin_time = pd.to_datetime(config_data["begin_time"], infer_datetime_format=True)
    end_time = pd.to_datetime(config_data["end_time"], infer_datetime_format=True)
    # Train or test, we can confirm that by folder name
    folder_to_save = input_path.split("/")[-1]
    folder_path = os.path.join(output_path, folder_to_save)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for dirname, _, filenames in os.walk(input_path):

        if filenames:
            # Last folder name = well id (TMP)
            well_id = dirname.split("/")[-1]
            well_path = os.path.join(folder_path, well_id)

            if not os.path.exists(well_path):
                os.makedirs(well_path)
        print(filenames)
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            print(f"Working with:\n{file_path}")
            data_file = pd.read_csv(file_path, parse_dates=["time"])
            data_file = data_file.sort_values(by="time")
            data_file = data_file.drop_duplicates(subset=["time"])
            cols = data_file.columns
            print("Dropped duplicates\nStarting calculating new index")
            if "time" not in cols:
                continue

            data_file = data_file.set_index("time")
            data_file = data_file[data_file.index <= end_time]
            new_idx = pd.date_range(begin_time, end_time, freq="2min")
            print("Starting interpolation")
            data_file = data_file.reindex(new_idx, method="nearest", limit=1).interpolate(method="spline", order=2)
            data_file = data_file.reset_index().rename(columns={"index": "time"})

            save_path = os.path.join(well_path, filename)
            print(f"Saved to\n{save_path}")
            data_file.to_csv(save_path)


if __name__ == "__main__":
    resample_data()

