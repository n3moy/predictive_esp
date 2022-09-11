import os

import yaml
import click
import pandas as pd


config_path = os.environ["CONFIG_PATH_PARAMS"]


@click.command()
@click.argument("featured_path", type=click.Path())
@click.argument("expanded_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def merge_features(
    featured_path: str,
    expanded_path: str,
    output_path: str
) -> None:
    task = featured_path.split("/")[-1]
    featured_filenames = os.listdir(featured_path)
    expanded_filenames = os.listdir(expanded_path)
    os.makedirs(output_path, exist_ok=True)

    for f_filename, e_filename in zip(featured_filenames, expanded_filenames):
        f_file_path = os.path.join(featured_path, f_filename)
        e_file_path = os.path.join(expanded_path, e_filename)
        f_data_file = pd.read_csv(f_file_path, parse_dates=["time"])
        e_data_file = pd.read_csv(e_file_path, parse_dates=["time"])

        assert f_data_file.shape[0] == e_data_file.shape[0]

        f_data_file["target_failure"] = e_data_file["failure_target"]
        f_data_file["event_id"] = e_data_file["event_id"]

        filename = "full_merged_" + e_filename[-5:]
        save_path = os.path.join(output_path, filename)
        f_data_file.to_csv(save_path, index=False)

    if task == "train" and featured_filenames:
        # Register necessary columns for resulting dataset
        config = yaml.safe_load(open(config_path))
        config["create_dataset"]["columns"] = f_data_file.columns.to_list()

        with open(config_path, "w") as f:
            yaml.dump(config, f, encoding="UTF-8", allow_unicode=True, default_flow_style=False)


if __name__ == "__main__":
    merge_features()
