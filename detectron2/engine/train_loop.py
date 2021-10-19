# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import time
import weakref
import torch
import random
import os
import statistics

from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def update_teacher(self, alpha=0.9):
        for teacher_param_name, teacher_param, student_param_name, student_param in \
                zip(self.base_model.named_parameters(), self.base_model.parameters(),
                    self.model.named_parameters(), self.model.parameters()):
            try:
                teacher_param.data.copy_(alpha * teacher_param.data + (1.0 - alpha) * student_param.data)
            except:
                pass
                # logger = logging.getLogger(__name__)
                # logger.info("Exception occured while updating teacher with param "
                #             + teacher_param_name[0] + ' ( shape: ' + str(teacher_param_name[1].shape) + ') from '
                #             + student_param_name[0] + ' ( shape: ' + str(student_param_name[1].shape) + ')')

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                t = []
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    start = time.perf_counter()
                    self.run_step()
                    if self.cfg.DISTILL.MEAN_TEACHER:
                        self.update_teacher(alpha=self.cfg.DISTILL.MEAN_TEACHER_ALPHA)
                    time_for_one_step = time.perf_counter() - start
                    self.after_step()
                    t.append(time_for_one_step)
                logger = logging.getLogger(__name__)
                logger.info('Average time for all iterations: ' + str(statistics.mean(t)))
                logger.info('Variance for all iterations: ' + str(statistics.variance(t)))
                logger.info('Min time for all iterations: ' + str(min(t)))
                logger.info('Max time for all iterations: ' + str(max(t)))
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        if comm.is_main_process():
            file_path = os.path.join(self.cfg.WG.IMAGE_STORE_LOC)
            if self.image_store is not None:
                with PathManager.open(file_path, "wb") as f:
                    torch.save(self.image_store, f)
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def update_image_store(self, images):
        for image in images:
            gt_classes = image["instances"].gt_classes
            cls = gt_classes[random.randrange(0, len(gt_classes))]
            self.image_store.add((image,), (cls,))

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        if (self.iter + 1) % self.cfg.WG.TRAIN_WARP_AT_ITR_NO == 0 and self.cfg.WG.ENABLE:
            verbose = False
            if verbose:
                logger = logging.getLogger(__name__)
                logger.info('Image store contains %d items. They are %s' % (len(self.image_store), self.image_store))

            self.cfg.WG.TRAIN_WARP = True

            self.optimizer.zero_grad()
            images = self.image_store.retrieve()

            if not self.cfg.WG.USE_FEATURE_STORE:
                for i in range(0, len(images), self.cfg.WG.BATCH_SIZE):
                    batched_images = images[i:i+self.cfg.WG.BATCH_SIZE]
                    loss_dict = self.model(batched_images)
                    cls_wrp = loss_dict.pop('loss_cls')
                    reg_wrp = loss_dict.pop('loss_box_reg')
                    warp_loss = cls_wrp + reg_wrp
                    self.optimizer.zero_grad()
                    warp_loss.backward()
                    for name, param in self.model.named_parameters():
                        if name not in self.cfg.WG.WARP_LAYERS and param.grad is not None:
                            param.grad.fill_(0)
                    self.optimizer.step()
            else:
                warp_loss_dict = self.model(images)
                warp_loss = sum(loss for loss in warp_loss_dict.values())
                self._detect_anomaly(warp_loss, warp_loss_dict)
                self.optimizer.zero_grad()
                warp_loss.backward()
                self.optimizer.step()

            self.cfg.WG.TRAIN_WARP = False

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        if self.image_store is not None:
            metrics_dict["num_images_in_ImageStore"] = len(self.image_store)
        self._write_metrics(metrics_dict)

        task_loss = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(task_loss, loss_dict)

        if self.cfg.WG.ENABLE:
            # Store the present data for future warp updates
            self.update_image_store(data)

            # Update task parameters on the task loss
            self.optimizer.zero_grad()
            task_loss.backward()
            for name, param in self.model.named_parameters():
                if name in self.cfg.WG.WARP_LAYERS:
                        param.grad.fill_(0)
            self.optimizer.step()
        elif self.cfg.FINETUNE.ENABLE:
            # Update only the task layers
            self.optimizer.zero_grad()
            task_loss.backward()
            for name, param in self.model.named_parameters():
                if name in self.cfg.WG.WARP_LAYERS:
                        param.grad.fill_(0)
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            task_loss.backward()
            self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            if 'num_images_in_ImageStore' in all_metrics_dict[0]:
                num_img = np.max([x.pop("num_images_in_ImageStore") for x in all_metrics_dict])
                self.storage.put_scalar("num_images_in_ImageStore", num_img)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
