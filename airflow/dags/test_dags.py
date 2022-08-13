import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "n3moy",
    "start_date": days_ago(0),
    "retries": 3,
    "task_concurrency": 1
}

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/params_all.yaml")

pipelines = {
    "train": {"schedule": "5 * * * *"},
    "predict": {"schedule": "* * * * *"},
    "build_features": {"schedule": "5 * * * *"},
    "create_dataset": {"schedule": "5 * * * *"}
}


def init_dag(dag, task_id):
    config = yaml.safe_load(open(CONFIG_PATH))[task_id]
    src_dir = config["src_dir"]
    CLI_params = config["CLI_params"]
    with dag:
        t1 = BashOperator(
            task_id=f"{task_id}",
            bash_command=f"python3 {src_dir} {CLI_params}"
        )
    return dag


for task_id, params in pipelines.items():

    dag = DAG(task_id,
              schedule_interval=params['schedule'],
              max_active_runs=1,
              default_args=default_args
              )
    init_dag(dag, task_id)
    globals()[task_id] = dag
