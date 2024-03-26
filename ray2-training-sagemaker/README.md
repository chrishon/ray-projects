# Reinforcement learning with Ray 2.x on SageMaker

The project assumes [Python3](https://www.python.org/downloads/) is available with [pip](https://pip.pypa.io/en/stable/installation/).

Create a virtual environment.
```bash
$ python3 -m venv env
```

Activate the virtual environment.
```bash
$ source env/bin/activate
```

Install python dependencies.
```bash
python -m pip install "sagemaker>=2.208.0"
```

Update the variable `role` in the script `start_sagemaker_training.py` so that it contains a valid [SageMaker execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) from your AWS account.
```python
role = "SMFullAccessRole"
```

Start the SageMaker training job.
```bash
python start_sagemaker_training.py
```
