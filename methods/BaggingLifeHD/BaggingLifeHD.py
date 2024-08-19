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


class BaggingLifeHD(LifeHD):
    def __init__(self, opt, train_loader, val_loader, num_classes, model, logger, device, num_learners):
        super(BaggingLifeHD, self).__init__(opt, train_loader, val_loader, num_classes, model, logger, device)
        self.num_learners = num_learners
        self.ensemble = [LifeHD(opt, self._bootstrap_sample(train_loader), val_loader, num_classes, model, logger, device)
                         for _ in range(self.num_learners)]


    def _bootstrap_sample(self, data_loader):
        dataset = data_loader.dataset
        indices = torch.randperm(len(dataset)).tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        new_data_loader = DataLoader(subset, batch_size=data_loader.batch_size, 
                                shuffle=True, 
                                num_workers=data_loader.num_workers)
    
        return new_data_loader
    

    
    def start(self):
        for model in self.ensemble:
            print("Starting Model!!!")
            print("Training Model!!!")
            model.train(1)
        print("Calling bag validate")
        self.validate(1, len(self.train_loader), True, 'final')
    
    # def validate(self, epoch, loader_idx, plot, mode):
    #     print("VALIDATING!!!")
    #     all_scores = []
    #     all_test_labels = []
    #     for model in self.ensemble:
    #         with torch.no_grad():
    #             scores = []
    #             for images, labels in tqdm(model.val_loader, desc="Testing"):
    #                 images = images.to(model.device)
    #                 outputs, _ = model.model(images)
    #                 scores.append(outputs.detach().cpu().numpy())
    #             all_scores.append(np.array(scores, dtype = object))
    #             all_test_labels.append(np.array([label.cpu().numpy() for _, label in model.val_loader], dtype=object))

    #     # Compute the final prediction by averaging scores and taking the argmax
    #     averaged_scores = np.mean(np.array(all_scores), axis=0)
    #     print(f"Shape of averaged_scores: {averaged_scores.shape}")
    #     majority_vote = np.argmax(averaged_scores)
        
    #     # Flattening the labels
    #     flat_test_labels = np.concatenate(all_test_labels[0])
        
    #     self._log_metrics(majority_vote, flat_test_labels, epoch, loader_idx, plot, mode)

    def validate(self, epoch, loader_idx, plot, mode):
        test_samples, test_embeddings = None, None
        pred_labels, test_labels = [], []
        
        # Initialize list to store predictions from each learner
        all_predictions = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Testing"):
                images = images.to(self.device)
                
                # Get predictions from each learner
                learner_predictions = []
                for learner in self.ensemble:
                    outputs, _ = learner(images)
                    predictions = torch.argmax(outputs, dim=-1)
                    learner_predictions.append(predictions.detach().cpu().tolist())
                
                # Aggregate predictions by majority voting
                learner_predictions = np.array(learner_predictions).T  # Shape: (num_samples, num_learners)
                aggregated_predictions = [np.bincount(preds).argmax() for preds in learner_predictions]
                
                # Gather aggregated prediction results
                pred_labels += aggregated_predictions
                test_labels += labels.cpu().tolist()

                # Gather raw samples and unnormalized embeddings
                embeddings = self.ensemble[0].encode(images).detach().cpu().numpy()
                test_bsz = images.shape[0]
                if test_embeddings is None:
                    test_samples = images.squeeze().view(
                        (test_bsz, -1)).cpu().numpy()
                    test_embeddings = embeddings
                else:
                    test_samples = np.concatenate(
                        (test_samples,
                        images.squeeze().view((test_bsz, -1)).cpu().numpy()),
                        axis=0)
                    test_embeddings = np.concatenate(
                        (test_embeddings, embeddings),
                        axis=0)

        # Convert lists to arrays
        pred_labels = np.array(pred_labels).astype(int)
        test_labels = np.array(test_labels).astype(int)
        
        # Log accuracy
        acc, purity, cm = eval_acc(test_labels, pred_labels)
        print('Acc: {}, purity: {}'.format(acc, purity))

        nmi = eval_nmi(test_labels, pred_labels)
        print('NMI: {}'.format(nmi))

        ri = eval_ri(test_labels, pred_labels)
        print('RI: {}'.format(ri))

    def _log_metrics(self, pred_labels, test_labels, epoch, loader_idx, plot, mode):
        acc, purity, cm = eval_acc(test_labels, pred_labels)
        print('Acc: {}, purity: {}'.format(acc, purity))
        nmi = eval_nmi(test_labels, pred_labels)
        print('NMI: {}'.format(nmi))
        ri = eval_ri(test_labels, pred_labels)
        print('RI: {}'.format(ri))

        with open(os.path.join(self.opt.save_folder, 'finalresult.txt'), 'a+') as f:
            f.write('{epoch},{idx},{acc},{purity},{nmi},{ri},{nc},{trim},{merge}\n'.format(
                epoch=epoch, idx=loader_idx, acc=acc, purity=purity,
                nmi=nmi, ri=ri, nc=self.model.cur_classes,
                trim=self.trim, merge=self.merge
            ))

        self.logger.log_value('accuracy', acc, loader_idx)
        self.logger.log_value('purity', purity, loader_idx)
        self.logger.log_value('nmi', nmi, loader_idx)
        self.logger.log_value('ri', ri, loader_idx)
        self.logger.log_value('num of clusters', self.model.cur_classes, loader_idx)