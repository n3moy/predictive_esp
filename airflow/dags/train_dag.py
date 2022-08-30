import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.external_task_sensor import ExternalTaskMarker, ExternalTaskSensor


default_args = {
    "owner": "n3moy",
    "start_date": days_ago(1),
    "retries": 1,
    "max_active_tis_per_dag'": 1,
    "trigger_rule": "all_success",
    "max_active_runs": 1,
    "parallelism": 1
}

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/cli_params.yaml")

pipelines = {
    "train": {"schedule": "20 * * * *"},
    "predict": {"schedule": "* * * * *"},
    "extract_data_dag": {"schedule": "5 * * * *"},
    "preprocess_dag": {"schedule": "5 * * * *"}
}


with DAG(
    "train_dag",
    default_args=default_args,
    description="From processed data to trained model",
    schedule_interval=pipelines["train"]["schedule"],
    catchup=False
) as train_dag:
    train_config = yaml.safe_load(open(CONFIG_PATH))["train"]
    sensor_train = ExternalTaskSensor(
        task_id="sensor_train",
        external_dag_id="preprocess_train_dag",
        external_task_id="maker_train_preprocess",
        allowed_states=["success", "skipped"],
        failed_states=["failed"],
        mode="reschedule"
    )
    t1_name = "train"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {train_config['src_dir']} {train_config['CLI_params']}"
    )

    maker_train1 = ExternalTaskMarker(
        task_id="maker_train1",
        external_dag_id="evaluate_dag",
        external_task_id="sensor_train_model"
    )
    # maker_train2 = ExternalTaskMarker(
    #     task_id="maker_train2",
    #     external_dag_id="evaluate_dag",
    #     external_task_id="sensor_predict"
    # )

    sensor_train >> t1 >> maker_train1

