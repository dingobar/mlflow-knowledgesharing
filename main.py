from datetime import datetime
from time import sleep, time
from pathlib import Path
import random

import mlflow


def do_smart_model_stuff():
    t0 = time()
    mlflow.log_param("param1", random.randint(0, 100))
    mlflow.log_param("start_datetime", datetime.now())

    mlflow.log_metric("foo", random.randint(0, 100))
    mlflow.log_dict({"foo": random.randint(0, 100)}, "metrics.json")

    with Path("output.txt").open("w") as f:
        f.write("Hello world!")

    mlflow.log_artifact("output.txt")

    sleep(random.randint(0, 10))

    mlflow.log_metric("run_time", time() - t0)

    mlflow.register_model("model", "model")


if __name__ == "__main__":
    mlflow.set_experiment("My Experiment")
    with mlflow.start_run():
        do_smart_model_stuff()
