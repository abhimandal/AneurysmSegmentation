import os, sys
import torch
import pyvista as pv
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

from tqdm.auto import tqdm
import time


class Trainer:
    def __init__(
        self, model, dataset, num_epoch=5, device=torch.device("cpu")
    ):
        self.num_epoch = num_epoch
        self._model = model
        self._dataset = dataset
        self.device = device

    def fit(self):
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=0.001
        )
        self.tracker = self._dataset.get_tracker(False, True)

        for i in range(self.num_epoch):
            print("=========== EPOCH %i ===========" % i)
            time.sleep(0.5)
            self.train_epoch()
            self.tracker.publish(i)
            self.test_epoch()
            self.tracker.publish(i)

    def train_epoch(self):
        self._model.to(self.device)
        self._model.train()
        self.tracker.reset("train")
        train_loader = self._dataset.train_dataloader
        iter_data_time = time.time()
        with tqdm(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                self.optimizer.zero_grad()
                data.to(self.device)
                self._model.forward(data)
                self._model.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    self.tracker.track(self._model)

                tq_train_loader.set_postfix(
                    **self.tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                )
                iter_data_time = time.time()

    def test_epoch(self):
        self._model.to(self.device)
        self._model.eval()
        self.tracker.reset("test")
        test_loader = self._dataset.test_dataloaders[0]
        iter_data_time = time.time()
        with tqdm(test_loader) as tq_test_loader:
            for i, data in enumerate(tq_test_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                data.to(self.device)
                self._model.forward(data)
                self.tracker.track(self._model)

                tq_test_loader.set_postfix(
                    **self.tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                )
                iter_data_time = time.time()
