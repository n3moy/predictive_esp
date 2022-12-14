import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.external_task_sensor import ExternalTaskMarker, ExternalTaskSensor
from dotenv import load_dotenv


default_args = {
    "owner": "n3moy",
    "start_date": days_ago(1),
    "retries": 1,
    "max_active_tis_per_dag'": 1,
    "trigger_rule": "all_success",
    "max_active_runs": 1,
    "parallelism": 1
}

load_dotenv()
config_path = os.getenv("CONFIG_PATH_CLI")


with DAG(
    "extract_data_dag",
    default_args=default_args,
    description="From raw data to renamed and split",
    schedule_interval="30 * * * *",
    catchup=False
) as extract_data_dag:
    tasks_params = yaml.safe_load(open(config_path))

    t1_name = "train_test_split"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {tasks_params[t1_name]['src_dir']} {tasks_params[t1_name]['CLI_params']}"
    )
    maker_train_extract = ExternalTaskMarker(
        task_id="maker_train_extract",
        external_dag_id="preprocess_train_dag",
        external_task_id="train_extracted"
    )
    maker_predict_extract = ExternalTaskMarker(
        task_id="maker_predict_extract",
        external_dag_id="preprocess_predict_dag",
        external_task_id="test_extracted"
    )

    maker_train_extract.set_upstream(t1)
    maker_predict_extract.set_upstream(t1)

