import os

import click
import pandas as pd

EVENTS_FILENAMES = {"1": "eventsData1.csv", "2": "eventsData2.csv", "3": "eventsData3.csv"}


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("events_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def join_events_to_data(
    input_path: str,
    events_path: str,
    output_path: str
) -> None:
    """
    This function assigns multiple events as marks into dataset based on startDate and endDate in events dataframe

    """
    # Train or test, we can confirm that by folder name
    folder_to_save = input_path.split("/")[-1]
    folder_path = os.path.join(output_path, folder_to_save)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path, parse_dates=["time"])
        data_file = data_file.set_index("time")
        well_id = filename[-5]
        events_file_path = os.path.join(events_path, EVENTS_FILENAMES[well_id])
        events_data = pd.read_csv(events_file_path, parse_dates=["startDate", "endDate"])

        events_dates = events_data[["startDate", "endDate"]].values
        events_id = events_data["result"].values
        data_file["event_id"] = 0

        for ev_id, (start_date, end_date) in zip(events_id, events_dates):
            mask = (data_file.index >= start_date) & (data_file.index <= end_date)
            data_file.loc[mask, "event_id"] = ev_id

        new_name = filename[:-5] + f"events_{well_id}.csv"
        save_path = os.path.join(folder_path, new_name)
        data_file.to_csv(save_path)


if __name__ == "__main__":
    join_events_to_data()

