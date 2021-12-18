from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task
import os


model_snapshots_path = os.path.join(os.getcwd(), 'clearml')
if not os.path.exists(model_snapshots_path):
    os.makedirs(model_snapshots_path)
task = Task.init(project_name="hrr",
                 task_name="optimization",
                 task_type=Task.TaskTypes.optimizer)

optimizer = HyperParameterOptimizer(
    base_task_id="000000000000000000000001",
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('number_of_epochs', min_value=2, max_value=12, step_size=2),
        UniformIntegerParameterRange('batch_size', min_value=2, max_value=16, step_size=2),
        UniformParameterRange('dropout', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('base_lr', min_value=0.00025, max_value=0.01, step_size=0.00025),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='accuracy',
    objective_metric_series='total',
    objective_metric_sign='max',

    # setting optimizer
    optimizer_class=OptimizerOptuna,

    # Configuring optimization parameters
    execution_queue='dan_queue',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=60.,
    compute_time_limit=120,
    total_max_jobs=20,
    min_iteration_per_job=15000,
    max_iteration_per_job=150000,
)