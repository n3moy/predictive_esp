import os

import click
import pandas as pd


@click.command()
@click.argument("featured_path", type=click.PATH())
@click.argument("expanded_path", type=click.PATH())
@click.argument("output_path", type=click.PATH())
def merge_features(
    featured_path: str,
    expanded_path: str,
    output_path: str
) -> None:
    folder_to_save = featured_path.split("/")[-1]
    folder_path = os.path.join(output_path, folder_to_save)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    featured_filenames = os.listdir(featured_path)
    expanded_filenames = os.listdir(expanded_path)

    for f_filename, e_filename in zip(featured_filenames, expanded_filenames):
        f_file_path = os.path.join(featured_path, f_filename)
        e_file_path = os.path.join(expanded_path, e_filename)
        f_data_file = pd.read_csv(f_file_path, parse_dates=["time"])
        e_data_file = pd.read_csv(e_file_path, parse_dates=["time"])

        assert f_data_file.shape[0] == e_data_file.shape[0]

        f_data_file["target_failure"] = e_data_file["failure_target"]
        f_data_file["event_id"] = e_data_file["event_id"]

        filename = "full_merged_" + e_filename[-5:]
        save_path = os.path.join(folder_path, filename)
        f_data_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    merge_features()
