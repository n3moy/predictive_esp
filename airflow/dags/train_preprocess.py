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
    "max_active_tis_per_dag": 1,
    "trigger_rule": "all_success",
    "max_active_runs": 1,
    "parallelism": 1
}

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/cli_params.yaml")
STRATEGY = "train"


with DAG(
    "preprocess_train_dag",
    default_args=default_args,
    description="From renamed data to processed train data",
    schedule_interval="30 * * * *",
    catchup=False
) as preprocess_train_dag:
    tasks_params = yaml.safe_load(open(CONFIG_PATH))["train_dag"]

    sensor_train_preprocess = ExternalTaskSensor(
        task_id="sensor_train_preprocess",
        external_dag_id="extract_data_dag",
        external_task_id="maker_train_extract",
        allowed_states=["success", "skipped"],
        failed_states=["failed"],
        mode="reschedule"
    )

    t1_name = "resample_data"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {tasks_params[t1_name]['src_dir']} {tasks_params[t1_name]['CLI_params']}",
    )
    t2_name = "merge_by_well"
    t2 = BashOperator(
        task_id=t2_name,
        bash_command=f"python3 {tasks_params[t2_name]['src_dir']} {tasks_params[t2_name]['CLI_params']}",
    )
    t3_name = "join_events"
    t3 = BashOperator(
        task_id=t3_name,
        bash_command=f"python3 {tasks_params[t3_name]['src_dir']} {tasks_params[t3_name]['CLI_params']}",
    )
    t4_name = "expand_target"
    t4 = BashOperator(
        task_id=t4_name,
        bash_command=f"python3 {tasks_params[t4_name]['src_dir']} {tasks_params[t4_name]['CLI_params']}",
    )
    t5_name = "build_features"
    t5 = BashOperator(
        task_id=t5_name,
        bash_command=f"python3 {tasks_params[t5_name]['src_dir']} {tasks_params[t5_name]['CLI_params']}",
    )
    t6_name = "merge"
    t6 = BashOperator(
        task_id=t6_name,
        bash_command=f"python3 {tasks_params[t6_name]['src_dir']} {tasks_params[t6_name]['CLI_params']}",
    )
    t7_name = "create_dataset"
    t7 = BashOperator(
        task_id=t7_name,
        bash_command=f"python3 {tasks_params[t7_name]['src_dir']} {tasks_params[t7_name]['CLI_params']}",
    )
    t8_name = "clear_nulls"
    t8 = BashOperator(
        task_id=t8_name,
        bash_command=f"python3 {tasks_params[t8_name]['src_dir']} {tasks_params[t8_name]['CLI_params']}",
    )
    maker_train_nulls = ExternalTaskMarker(
        task_id="maker_train_nulls",
        external_dag_id="preprocess_predict_dag",
        external_task_id="sensor_predict_nulls"
    )
    maker_train_preprocess = ExternalTaskMarker(
        task_id="maker_train_preprocess",
        external_dag_id="train_dag",
        external_task_id="sensor_train"
    )

    t1.set_upstream(sensor_train_preprocess)
    t2.set_upstream(t1)
    t3.set_upstream(t2)
    t4.set_upstream(t3)
    t5.set_upstream(t2)
    t6.set_upstream(t5)
    t6.set_upstream(t4)
    t7.set_upstream(t6)
    t8.set_upstream(t7)
    # Signal to predict preprocess dag
    maker_train_nulls.set_upstream(t8)
    maker_train_preprocess.set_upstream(maker_train_nulls)



