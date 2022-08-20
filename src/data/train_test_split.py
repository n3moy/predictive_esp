import os
from random import randint

import click
import pandas as pd

TEST_FOLDER = "test"
TRAIN_FOLDER = "train"


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("window_size", type=click.INT)
def split(
    input_path: str,
    output_path: str,
    window_size: int
) -> None:
    test_output_path = os.path.join(output_path, TEST_FOLDER)
    train_output_path = os.path.join(output_path, TRAIN_FOLDER)
    start_test_month = randint(1, 12-window_size)

    for dirname, _, filenames in os.walk(input_path):
        try:
            save_folder = int(dirname[-1])
        except ValueError:
            continue

        save_folder = str(save_folder)
        test_folder_path = os.path.join(test_output_path, save_folder)
        train_folder_path = os.path.join(train_output_path, save_folder)

        print(f"Test folder path:\n{test_folder_path}")
        print(f"Train folder path:\n{train_folder_path}")

        os.makedirs(test_folder_path, exist_ok=True)
        os.makedirs(train_folder_path, exist_ok=True)

        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            data_file = pd.read_csv(file_path, parse_dates=["time"])
            data_file["month"] = data_file["time"].dt.month
            mask = (data_file["month"] >= start_test_month) & (data_file["month"] <= start_test_month + window_size)
            test_data = data_file.loc[mask, :]
            train_data = data_file.loc[~mask, :]
            test_data = test_data.drop("month", axis=1)
            train_data = train_data.drop("month", axis=1)
            test_data = test_data.reset_index(drop=True)
            train_data = train_data.reset_index(drop=True)

            test_save_path = os.path.join(test_folder_path, filename)
            train_save_path = os.path.join(train_folder_path, filename)
            train_data.to_csv(train_save_path, index=False)
            test_data.to_csv(test_save_path, index=False)


if __name__ == "__main__":
    split()
