# Imports
# from sagemaker.rl.estimator import RLEstimator
from sagemaker.estimator import Estimator as RLEstimator
from sagemaker import image_uris
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import (
    ContinuousParameter,
    CategoricalParameter,
    IntegerParameter,
)

from tuning_config import hyperparameter_tuning

TUNE = False

# NOTE: make sure to replace the role with an existing sagemaker execution role within your account
role = "SMFullAccessRole"

# Retrieve the required tensorflow container image
instance_type = "ml.m5.large"
image_uri = image_uris.retrieve(
    framework="pytorch",
    region="eu-central-1",
    version="2.0",
    py_version="py310",
    image_scope="training",
    instance_type=instance_type,
)

# Metrics definition to visualize metrics within SageMaker dashboards
float_regex = "[-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?"
metric_definitions = [
    {"Name": "episode_reward_mean", "Regex": r"episode_reward_mean\s*(%s)" % float_regex},
    {"Name": "episode_reward_max", "Regex": r"episode_reward_max\s*(%s)" % float_regex},
]

# The actual estimator
estimator = RLEstimator(
    entry_point="train-rl-cartpole-ray.py",
    source_dir="src",
    image_uri=image_uri,
    role=role,
    debugger_hook_config=False,
    instance_type=instance_type,
    instance_count=1,
    base_job_name="rl-cartpole-ray-2x",
    metric_definitions=metric_definitions,
    hyperparameters={
        # Let's override some hyperparameters
        "rl.training.config.lr": 0.0001,
    },
)

# Training start
if not TUNE:
    estimator.fit(wait=False)
    job_name = estimator.latest_training_job.job_name
    print("Training job: %s" % job_name)
else:
    hyperparameter_ranges = {}
    for key, val in hyperparameter_tuning.get("hyperparameter_ranges", {}).items():
        if type(val) is dict:
            if type(val["min_value"]) is int:
                
                hyperparameter_ranges[key] = IntegerParameter(
                    min_value=val["min_value"], max_value=val["max_value"]
                )
            else:
                hyperparameter_ranges[key] = ContinuousParameter(
                    min_value=val["min_value"], max_value=val["max_value"]
                )
        if type(val) is list:
            hyperparameter_ranges[key] = CategoricalParameter(val)
            
    tuner = HyperparameterTuner(
                    estimator=estimator,
                    objective_metric_name=hyperparameter_tuning.get("objective_metric", "episode_reward_mean"),
                    hyperparameter_ranges=hyperparameter_ranges,
                    metric_definitions=metric_definitions,
                    max_jobs=hyperparameter_tuning.get("max_jobs", 1),
                    max_parallel_jobs=hyperparameter_tuning.get("max_parallel_jobs", 1),
                )

