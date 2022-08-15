import os

import click
import pandas as pd
import yaml

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/params_all.yaml")
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def join_data(
    input_path: str,
    output_path: str,
    # update_file: str
) -> None:
    """
    This function joins all datasets in 'list_data' into one pd.DataFrame and splits it in train and test
    """
    print("BEGIN" + "*"*8)
    list_data = os.listdir(input_path)
    config = yaml.safe_load(open(CONFIG_PATH))["create_dataset"]
    data_cols = config["columns"]
    joined_df = pd.DataFrame(columns=data_cols)
    print("BEGIN CYCLE" + "*" * 8)
    for df_name in list_data:
        df_path = os.path.join(input_path, df_name)
        df = pd.read_csv(df_path)
        joined_df = pd.concat([joined_df, df], axis=0)
    print("END CYCLE" + "*" * 8)
    joined_df = joined_df.reset_index(drop=True)
    joined_df["time"] = pd.to_datetime(joined_df["time"])
    joined_df = joined_df.sort_values(by="time", ascending=True)
    # joined_df = pd.concat(
    #     [joined_df, pd.get_dummies(joined_df["well"], prefix="Well")], axis=1
    # )
    # joined_df = joined_df.drop("well", axis=1)
    # joined_df = joined_df.reset_index(drop=True)

    test_df = joined_df[
        (
            (joined_df["time"].dt.date >= pd.Timestamp(f"2021-06-01"))
            & (joined_df["time"].dt.date <= pd.Timestamp(f"2021-06-30"))
        )
    ]
    print(f"Joined data shape : {joined_df.shape}")
    test_indices = test_df.index
    train_df = joined_df.drop(test_indices, axis=0)
    # train_df = train_df.loc[
    #   ((train_df["stable"] == 1) & (train_df["failure_target"] == 0)) | ((train_df["failure_target"] == 1))
    # ]
    test_df = test_df.drop(["stable", "time_to_failure"], axis=1)
    train_df = train_df.drop(["stable", "time_to_failure"], axis=1)

    test_save_path = os.path.join(output_path, TEST_FILENAME)
    train_save_path = os.path.join(output_path, TRAIN_FILENAME)
    test_df.to_csv(test_save_path, index=False)
    train_df.to_csv(train_save_path, index=False)

    # with open(update_file, "a") as f:
    #     upd_date = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    #     f.write(f"\nUpdate at {upd_date}")


if __name__ == "__main__":
    join_data()
