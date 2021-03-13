import os
from logging import WARNING
import warnings
warnings.filterwarnings("ignore")

import psutil
import ray

from logger import Logger
from utils import set_seed, save_metrics_params, update_params_from_cmdline, save_settings_to_json

import warcraft_shortest_path.data_utils as warcraft_shortest_path_data
import warcraft_shortest_path.trainers as warcraft_shortest_path_trainers

dataset_loaders = {
    "warcraft_shortest_path": warcraft_shortest_path_data.load_dataset
}

trainer_loaders = {
    "warcraft_shortest_path": warcraft_shortest_path_trainers.get_trainer
}

required_top_level_params = [
    "model_dir",
    "seed",
    "loader_params",
    "problem_type",
    "trainer_name",
    "trainer_params",
    "num_epochs",
    "evaluate_every",
    "save_visualizations"
]
optional_top_level_params = ["num_cpus", "use_ray", "default_json", "id", "fast_mode", "fast_forward_training"]

def verify_top_level_params(**kwargs):
    for kwarg in kwargs:
        if kwarg not in required_top_level_params and kwarg not in optional_top_level_params:
            raise ValueError("Unknown top_level argument: {}".format(kwarg))

    for required in required_top_level_params:
        if required not in kwargs.keys():
            raise ValueError("Missing required argument: {}".format(required))

def main():
    params = update_params_from_cmdline(verbose=True)
    os.makedirs(params.model_dir, exist_ok=True)
    save_settings_to_json(params, params.model_dir)

    num_cpus = params.get("num_cpus", psutil.cpu_count(logical=True))
    use_ray = params.get("use_ray", False)
    fast_forward_training = params.get("fast_forward_training", False)
    if use_ray:
        ray.init(
            num_cpus=num_cpus,
            logging_level=WARNING,
            ignore_reinit_error=True,
            redis_max_memory=10 ** 9,
            log_to_driver=False,
            **params.get("ray_params", {})
        )

    set_seed(params.seed)

    Logger.configure(params.model_dir, "tensorboard")

    dataset_loader = dataset_loaders[params.problem_type]
    train_iterator, test_iterator, metadata = dataset_loader(**params.loader_params)

    trainer_class = trainer_loaders[params.problem_type](params.trainer_name)

    fast_mode = params.get("fast_mode", False)
    trainer = trainer_class(
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        metadata=metadata,
        fast_mode=fast_mode,
        **params.trainer_params
    )
    train_results = {}
    for i in range(params.num_epochs):
        if i % params.evaluate_every == 0:
            eval_results = trainer.evaluate()
            print(eval_results)

        train_results = trainer.train_epoch()
        if train_results["train_accuracy"] > 0.999 and fast_forward_training:
            print(f'Reached train accuracy of {train_results["train_accuracy"]}. Fast forwarding.')
            break


    eval_results = trainer.evaluate()
    print(eval_results)
    train_results = train_results or {}
    save_metrics_params(params=params, metrics={**eval_results, **train_results})

    if params.save_visualizations:
        print("Saving visualization images")
        trainer.log_visualization()

    if use_ray:
        ray.shutdown()


if __name__ == "__main__":
    main()
