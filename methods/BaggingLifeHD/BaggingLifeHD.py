from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms.functional import rotate
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from tqdm import tqdm
from utils.eval_utils import eval_acc, eval_nmi, eval_ri
from utils.plot_utils import plot_tsne, plot_tsne_graph, \
    plot_novelty_detection, plot_confusion_matrix
from methods.LifeHD.LifeHD import LifeHD 

import random

class BaggingLifeHD:
    def __init__(self, opt, train_loader, val_loader, num_classes, model, logger, device, num_models):
        """
        Initialize the Bagging ensemble of LifeHD models

        Args:
            num_models: Number of models in the ensemble
            opt: The options/parameters for the model
            train_loader: The data loader for training
            val_loader: The data loader for validation
            num_classes: The number of classes in the dataset
            model_class: The class of the model to use (LifeHD in this case)
            logger: Logger for tensorboard or other logging
            device: The device to use (CPU or GPU)
        """
        self.num_models = num_models
        self.models = []
        self.opt = opt

        # Create and train num_models instances of LifeHD
        for i in range(num_models):
            # Create a bootstrap sample of the training data
            bootstrap_train_loader = self.create_bootstrap_loader(train_loader)

            # Initialize a new model
            model_instance = LifeHD(opt, bootstrap_train_loader, val_loader, num_classes, 
                                         model, logger, device)

            # Train the model on the bootstrap sample
            model_instance.start()
            self.models.append(model_instance)

    def create_bootstrap_loader(self, original_loader):
        """
        Create a bootstrap sample from the original data loader

        Args:
            original_loader: Original data loader

        Returns:
            A data loader containing the bootstrap sample
        """
        bootstrap_dataset = []
        dataset_size = len(original_loader.dataset)
        indices = random.choices(range(dataset_size), k=dataset_size)

        for idx in indices:
            bootstrap_dataset.append(original_loader.dataset[idx])

        # Create a new DataLoader using the bootstrap dataset
        return torch.utils.data.DataLoader(bootstrap_dataset, batch_size=original_loader.batch_size, shuffle=True)

    def aggregate_predictions(self, images):
        """
        Aggregate predictions from all models in the ensemble

        Args:
            images: The input images for which predictions are required

        Returns:
            The aggregated predictions
        """
        predictions = []
        
        for model in self.models:
            outputs, _ = model.model(images)
            predictions.append(outputs)

        # Aggregate the predictions (e.g., majority voting)
        final_prediction = torch.mean(torch.stack(predictions), dim=0)
        return torch.argmax(final_prediction, dim=-1)

    def validate(self):
        """
        Validate the ensemble on the validation set
        """
        pred_labels, test_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(self.models[0].val_loader, desc="Testing Ensemble"):
                images = images.to(self.opt.device)

                # Aggregate predictions across all models
                predictions = self.aggregate_predictions(images)

                pred_labels += predictions.detach().cpu().tolist()
                test_labels += labels.cpu().tolist()

        # Evaluate performance using eval_acc, eval_nmi, eval_ri
        acc, purity, cm = eval_acc(np.array(test_labels), np.array(pred_labels))
        print(f'Ensemble Acc: {acc}, purity: {purity}')

        nmi = eval_nmi(np.array(test_labels), np.array(pred_labels))
        print(f'Ensemble NMI: {nmi}')

        ri = eval_ri(np.array(test_labels), np.array(pred_labels))
        print(f'Ensemble RI: {ri}')

        return acc, purity, nmi, ri

    
