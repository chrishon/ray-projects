OBJECTIVE_METRIC = "episode_reward_mean"
MAX_JOBS = 2
MAX_PARALLEL_JOBS = 2

hyperparameter_ranges = {
                         "rl.training.config.lr":{"min_value":0.000001,"max_value":0.01},
                         }
 
hyperparameter_tuning = {
                         "hyperparameter_ranges": hyperparameter_ranges,
                                                   "objective_metric":OBJECTIVE_METRIC,
                                                   "max_jobs":MAX_JOBS,
                                                   "max_parallel_jobs":MAX_PARALLEL_JOBS}