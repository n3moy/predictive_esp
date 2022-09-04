import os

import click
import yaml
import joblib
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from dotenv import load_dotenv


def evaluate_results(
    data_in: pd.DataFrame,
    forecast_window: int = 14
) -> pd.DataFrame:
    data = data_in.copy()
    cols = data.columns

    if "time" not in cols:
        raise KeyError("'Time' column is not found in data")

    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values(by="time", ascending=True)
    data.loc[(data["event_id"] != 0), "failure_date"] = data.loc[(data["event_id"] != 0), "time"]
    data["failure_date"] = data["failure_date"].bfill().fillna(data["time"].max())
    data["failure_date"] = pd.to_datetime(data["failure_date"])

    data["Y_FAIL_sumxx"] = 0
    data["Y_FAIL_sumxx"] = (data["predicted"].rolling(min_periods=1, window=forecast_window).sum())

    # if a signal has occured in the last 14 days, the signal is 0.
    data["Y_FAILZ"] = np.where((data.Y_FAIL_sumxx > 1), 0, data.predicted)
    # sort the data by id and date.
    data = data.sort_values(by=["time"], ascending=[True])

    # create signal id with the cumsum function.
    data["SIGNAL_ID"] = data["Y_FAILZ"].cumsum()
    df_signals = data[data["Y_FAILZ"] == 1].copy()
    df_signal_date = df_signals[["SIGNAL_ID", "time"]].copy()
    df_signal_date = df_signal_date.rename(index=str, columns={"time": "SIGNAL_DATE"})
    data = data.merge(df_signal_date, on=["SIGNAL_ID"], how="outer")

    data["C"] = data["failure_date"] - data["SIGNAL_DATE"]
    data["WARNING"] = data["C"] / np.timedelta64(1, "D")

    data["true_failure"] = data["event_id"] != 0
    data["FACT_FAIL_sumxx"] = 0
    data["FACT_FAIL_sumxx"] = (data["true_failure"].rolling(min_periods=1, window=forecast_window).sum())

    # if a signal has occured in the last 90 days, the signal is 0.
    data["actual_failure"] = np.where((data.FACT_FAIL_sumxx > 1), 0, data.true_failure)

    # define a true positive
    data["TRUE_POSITIVE"] = np.where(
        ((data.actual_failure == 1) & (data.WARNING <= forecast_window) & (data.WARNING >= 0)), 1, 0)
    # define a false negative
    data["FALSE_NEGATIVE"] = np.where((data.TRUE_POSITIVE == 0) & (data.actual_failure == 1), 1, 0)
    # define a false positive
    data["BAD_S"] = np.where((data.WARNING < 0) | (data.WARNING >= forecast_window), 1, 0)
    data["FALSE_POSITIVE"] = np.where(((data.Y_FAILZ == 1) & (data.BAD_S == 1)), 1, 0)
    data["bootie"] = 1
    data["CATEGORY"] = np.where((data.FALSE_POSITIVE == 1), "FALSE_POSITIVE",
                                (np.where((data.FALSE_NEGATIVE == 1), "FALSE_NEGATIVE",
                                          (np.where((data.TRUE_POSITIVE == 1), "TRUE_POSITIVE", "TRUE_NEGATIVE")))))
    table = pd.pivot_table(data, values=["bootie"], columns=["CATEGORY"], aggfunc=np.sum)

    return table


def get_version_model(config_name, client):
    """
    Gets last version from MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        dict_push[count] = value
    print(dict_push)
    return dict(list(dict_push.items())[-1][1])["version"]


load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
config_path = os.getenv("CONFIG_PATH_PARAMS")
FILENAME = "evaluate_preds.csv"


@click.command()
@click.argument("model_path", type=click.Path())
@click.argument("data_path", type=click.Path())
@click.argument("target_name", type=click.STRING)
@click.argument("output_path", type=click.Path())
def evaluate(
    model_path: str,
    data_path: str,
    target_name: str,
    output_path: str
) -> None:

    test_data = pd.read_csv(data_path, parse_dates=["time"])
    # time_series = test_data["time"]
    # test_data = test_data.drop("time", axis=1)
    test_data = test_data.set_index("time")
    test_data = test_data.dropna()
    model = joblib.load(model_path)
    X, y = test_data.drop([target_name, "event_id"], axis=1), test_data[target_name]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow_experiment_id = "lr_model"
    mlflow.set_experiment(mlflow_experiment_id)
    config = yaml.safe_load(open(config_path))

    with mlflow.start_run():
        prediction = model.predict(X)
        accuracy = accuracy_score(y, prediction)
        precision = precision_score(y, prediction)
        recall = recall_score(y, prediction)
        roc_auc = roc_auc_score(y, prediction)
        f1 = f1_score(y, prediction)

        test_data["predicted"] = prediction
        test_data = test_data.reset_index()
        table_metrics = evaluate_results(test_data)
        table_metrics = table_metrics.to_dict(orient="records")[0]

        mlflow.log_metric("Accuracy_signal", accuracy)
        mlflow.log_metric("Precision_signal", precision)
        mlflow.log_metric("Recall_signal", recall)
        mlflow.log_metric("ROC_AUC_signal", roc_auc)
        mlflow.log_metric("F1_score_signal", f1)

        mlflow.log_metrics(table_metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lr_model",
            registered_model_name=config["train"]["model_name"]
        )

        mlflow.end_run()

    save_path = os.path.join(output_path, FILENAME)
    np.savetxt(save_path, prediction, delimiter=",")

    # Saving last version of a model
    client = MlflowClient()
    last_version_lr = get_version_model(config["train"]["model_name"], client)

    yaml_file = yaml.safe_load(open(config_path))
    yaml_file["evaluate"]["model_version"] = int(last_version_lr)

    with open(CONFIG_PATH, "w") as fp:
        yaml.dump(yaml_file, fp, encoding="UTF-8", allow_unicode=True)


if __name__ == "__main__":
    evaluate()


