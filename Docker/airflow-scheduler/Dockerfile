FROM apache/airflow:2.3.4-python3.8

USER root
RUN apt-get update && apt-get -y install gcc && apt-get -y install git
USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt