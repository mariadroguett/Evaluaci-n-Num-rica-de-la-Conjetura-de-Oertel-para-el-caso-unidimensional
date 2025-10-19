Bootstrap: docker
From: python:3.9-slim

%post
    apt-get update && apt-get install -y build-essential
    pip install --no-cache-dir -r /opt/app/requirements.txt

%files
    requirements.txt /opt/app/requirements.txt
    . /opt/app

%environment
    export PYTHONPATH=/opt/app:$PYTHONPATH
