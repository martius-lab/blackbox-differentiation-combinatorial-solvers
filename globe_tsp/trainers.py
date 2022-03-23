from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import random
import time
from abc import ABC, abstractmethod
import torch
from comb_modules.losses import HammingLoss
from comb_modules.tsp import TspSolver
from logger import Logger
from models import get_model
from utils import AverageMeter, optimizer_from_string, customdefaultdict
from collections import defaultdict
from . import metrics
import numpy as np
from data.utils import get_tsp_path_plot


def get_trainer(trainer_name):
    trainers = {"TSPApproximate": TSPApproximateTrainer}
    return trainers[trainer_name]


class TSPAbstractTrainer(ABC):
    def __init__(
            self,
            *,
            train_iterator,
            test_iterator,
            metadata,
            use_cuda,
            batch_size,
            optimizer_name,
            optimizer_params,
            model_params,
            fast_mode,
            preload_batch,
            lr_milestones,
            use_lr_scheduling
    ):

        self.fast_mode = fast_mode
        self.use_cuda = use_cuda
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.test_iterator = test_iterator
        self.train_iterator = train_iterator
        self.metadata = metadata
        self.preload_batch = preload_batch

        self.model = None
        self.build_model(**model_params)

        if self.use_cuda:
            self.model.to("cuda")
        self.optimizer = optimizer_from_string(optimizer_name)(self.model.parameters(), betas=(0.5, 0.999),
                                                               **optimizer_params)
        self.use_lr_scheduling = use_lr_scheduling
        if use_lr_scheduling:
            self.scheduler = MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.1)
            # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=1, threshold=0.005, min_lr=1e-9, verbose=True)
        self.epochs = 0
        self.train_logger = Logger(scope="training", default_output="tensorboard")
        self.val_logger = Logger(scope="validation", default_output="tensorboard")

    def train_epoch(self):
        self.model.train()
        self.epochs += 1

        batch_time = AverageMeter("Batch time")
        data_time = AverageMeter("Data time")
        avg_loss = AverageMeter("Loss")
        avg_accuracy = AverageMeter("Accuracy")
        avg_perfect_accuracy = AverageMeter("Perfect Accuracy")

        avg_metrics = customdefaultdict(lambda k: AverageMeter("train_" + k))

        end = time.time()

        iterator = self.train_iterator.get_epoch_iterator(batch_size=self.batch_size, number_of_epochs=1,
                                                          device='cuda' if self.use_cuda else 'cpu',
                                                          preload=self.preload_batch)
        for i, data in enumerate(iterator):
            flags, true_tours, true_distances = data["flags"], data["labels"], data["true_distances"]

            if i == 0:
                self.log(data, train=True)

            # measure data loading time
            data_time.update(time.time() - end)

            loss, accuracy, last_suggestion = self.forward_pass(flags, true_tours, train=True, i=i)

            suggested_tours = last_suggestion["suggested_tours"]

            # update batch metrics
            batch_metrics = metrics.compute_metrics(true_tours, suggested_tours, true_distances)
            for k, v in batch_metrics.items():
                avg_metrics[k].update(v, flags.size(0))

            assert len(avg_metrics.keys()) > 0

            avg_loss.update(loss.item(), flags.size(0))
            avg_accuracy.update(accuracy.item(), flags.size(0))
            avg_perfect_accuracy.update(batch_metrics['perfect_match_accuracy'], flags.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.fast_mode:
                break

        meters = [batch_time, data_time, avg_loss, avg_accuracy, avg_perfect_accuracy]
        meter_str = "\t".join([str(meter) for meter in meters])
        print(f"Epoch: {self.epochs}\t{meter_str}")

        if self.use_lr_scheduling:
            self.scheduler.step(self.epochs)
            # self.scheduler.step(avg_perfect_accuracy.avg, epoch=self.epochs)

        self.train_logger.log(avg_loss.avg, "loss")
        self.train_logger.log(avg_accuracy.avg, "accuracy")
        for key, avg_metric in avg_metrics.items():
            self.train_logger.log(avg_metric.avg, key=key)

        return {
            "train_loss": avg_loss.avg,
            "train_accuracy": avg_accuracy.avg,
            **{"train_" + k: avg_metrics[k].avg for k in avg_metrics.keys()}
        }

    def evaluate(self):
        self.model.eval()

        avg_metrics = defaultdict(AverageMeter)

        iterator = self.test_iterator.get_epoch_iterator(batch_size=self.batch_size, number_of_epochs=1, shuffle=False,
                                                         device='cuda' if self.use_cuda else 'cpu',
                                                         preload=self.preload_batch)

        for i, data in enumerate(iterator):
            input, true_path, true_distances = (
                data["flags"].contiguous(),
                data["labels"].contiguous(),
                data["true_distances"].contiguous(),
            )

            if self.use_cuda:
                input = input.cuda(async=True)
                true_path = true_path.cuda(async=True)

            loss, accuracy, last_suggestion = self.forward_pass(input, true_path, train=False, i=i)
            suggested_tours = last_suggestion["suggested_tours"]
            data.update(last_suggestion)
            if i == 0:
                indices_in_batch = random.sample(range(self.batch_size), 4)
                for num, k in enumerate(indices_in_batch):
                    self.log(data, train=False, k=k, num=num)

            evaluated_metrics = metrics.compute_metrics(true_path, suggested_tours, true_distances)

            avg_metrics["loss"].update(loss.item(), input.size(0))
            avg_metrics["accuracy"].update(accuracy.item(), input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])
        return avg_metrics_values

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def forward_pass(self, input, label, train, i):
        pass

    def log(self, data, train, k=None, num=None):
        logger = self.train_logger if train else self.val_logger
        if not train:
            suggested_tours = data["suggested_tours"][k].squeeze()
            labels = data["labels"][k].squeeze()
            flags = np.array(data["flags"][k]).transpose(0, 2, 3, 1).astype("uint8")
            coordinates = data["coordinates"][k].detach().numpy()

            suggested_tours_im = torch.ones((3, *suggested_tours.shape)) * 255 * suggested_tours.cpu()
            labels_im = torch.ones((3, *labels.shape)) * 255 * labels.cpu()
            image_with_path = get_tsp_path_plot(coordinates, flags, labels)

            logger.log(labels_im.data.numpy().astype(np.uint8), key=f"best_tour_{num}", data_type="image")
            logger.log(suggested_tours_im.data.numpy().astype(np.uint8), key=f"suggested_tour_{num}", data_type="image")
            logger.log(image_with_path, key=f"full_input_with_tour_{num}", data_type="image")


def get_unit_sphere_projections(coordinates):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    lengths = (x ** 2 + y ** 2 + z ** 2).sqrt()
    unit_coordinates = torch.empty_like(coordinates)
    for i, coord in enumerate(coordinates):
        unit_coordinates[i] = coord / lengths[i]
    return unit_coordinates


class TSPApproximateTrainer(TSPAbstractTrainer):

    def __init__(self, *, l1_regconst, lambda_val, approximation_params, **kwargs):
        super().__init__(**kwargs)
        self.l1_regconst = l1_regconst
        self.lambda_val = lambda_val
        self.solver = TspSolver(lambda_val, approximation_params)
        self.loss_fn = HammingLoss()
        print("META:", self.metadata)

    def build_model(self, model_name, arch_params):
        self.model = get_model(model_name, out_features=3, in_channels=self.metadata["num_channels"],
                               arch_params=arch_params)

    def forward_pass(self, flags, label, train, i):
        flags_flat = flags.reshape(flags.shape[0] * flags.shape[1], flags.shape[2], flags.shape[3], flags.shape[4])

        outputs = self.model(flags_flat)
        coordinates = get_unit_sphere_projections(outputs)
        coordinates = coordinates.reshape(flags.shape[0], flags.shape[1], 3)
        distance_matrices = torch.cdist(coordinates, coordinates, 2.0)

        if i == 0 and not train:
            print(coordinates[0])
        assert len(distance_matrices.shape) == 3, f"{str(distance_matrices.shape)}"

        tours = self.solver(distance_matrices)

        loss = self.loss_fn(tours, label)

        last_suggestion = {
            "suggested_distances": distance_matrices,
            "suggested_tours": tours,
            "coordinates": coordinates
        }

        accuracy = (torch.abs(tours - label) < 0.5).to(torch.float32).mean()
        extra_loss = self.l1_regconst * torch.mean(torch.abs(outputs))
        loss += extra_loss

        return loss, accuracy, last_suggestion
