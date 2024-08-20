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
        self.ensemble = [LifeHD(opt, self._bootstrap_sample(train_loader, _), val_loader, num_classes, model, logger, device)
                         for _ in range(self.num_learners)]


    def _bootstrap_sample(self, data_loader, seed):
        print("using seed: ", seed)
        dataset = data_loader.dataset
        torch.manual_seed(seed)
        indices = torch.randperm(len(dataset)).tolist()
        subset = torch.utils.data.Subset(dataset, indices[:len(indices)//10])
        print("SUBSET!!!")
        print(subset[0:10])
        print(torch.sum(subset))
        new_data_loader = DataLoader(subset, batch_size=data_loader.batch_size, 
                                shuffle=True, 
                                num_workers=data_loader.num_workers)
    
        return new_data_loader
    

    
    def start(self):
        for model in self.ensemble:
            print("Starting model: ", model)
            model.start()
        print("Starting Validation")
        self.validate(1, len(self.ensemble[0].train_loader), True, 'final')

    def validate(self, epoch, loader_idx, plot, mode):
        test_samples, test_embeddings = None, None
        pred_labels, test_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.ensemble[0].val_loader, desc="Testing"): #questionable
                images = images.to(self.ensemble[0].device)
                scores = []
                for model in self.ensemble:
                    outputs, _ = model.model(images)
                    if len(scores) == 0:
                        scores = outputs
                        print("first model")
                    else:
                        print("SCORES!!!")
                        print(scores)
                        print("OUTPUTS!!!")
                        print(outputs)
                        scores = (scores + outputs) / 2
                        #print(scores)
                        print("averaging scores")
                    #print(type(outputs))
                    #print(outputs)
                    #avg outputs with scores

                predictions = torch.argmax(scores,dim=-1)
                pred_labels += predictions.detach().cpu().tolist()
                test_labels += labels.cpu().tolist()

                #questionable
                embeddings = self.ensemble[0].model.encode(images).detach().cpu().numpy()
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
        
        # log accuracy
        pred_labels = np.array(pred_labels).astype(int)
        print(np.unique(pred_labels))
        test_labels = np.array(test_labels).astype(int)
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

        # tensorboard logger
        self.logger.log_value('accuracy', acc, loader_idx)
        self.logger.log_value('purity', purity, loader_idx)
        self.logger.log_value('nmi', nmi, loader_idx)
        self.logger.log_value('ri', ri, loader_idx)
        self.logger.log_value('num of clusters', self.model.cur_classes, loader_idx)

        # # plot raw and high-dimensional embeddings
        # if plot:
        #     # plot the tSNE of raw samples with predicted labels
        #     #plot_tsne(test_samples, np.array(pred_labels), np.array(test_labels),
        #     #          title='raw samples {} {} {}'.format(self.opt.method, self.opt.dataset, acc),
        #     #          fig_name=os.path.join(self.opt.save_folder,
        #     #                                '{}_sap_{}_{}.png'.format(
        #     #                                    loader_idx, self.opt.method, self.opt.dataset)))
        #     # plot the tSNE of embeddings with predicted labels
        #     plot_tsne(test_embeddings, np.array(pred_labels), np.array(test_labels),
        #               title='embeddings {} {} {} {}'.format(self.opt.method, self.opt.dataset, acc, mode),
        #               fig_name=os.path.join(self.opt.save_folder,
        #                                     '{}_emb_{}_{}_{}.png'.format(
        #                                         loader_idx, self.opt.method, self.opt.dataset, mode)))

        #     # plot embeddings with class hypervectors
        #     #class_hvs = self.model.extract_class_hv()  # numpy array
        #     #plot_tsne_graph(class_hvs,
        #     #                title='class hvs {} {} {}'.format(self.opt.method, self.opt.dataset, acc),
        #     #                fig_name=os.path.join(self.opt.save_folder,
        #     #                                      '{}_cls_hv_{}_{}.png'.format(
        #     #                                          loader_idx, self.opt.method, self.opt.dataset)))

        #     # save confusion matrix
        #     np.save(os.path.join(self.opt.save_folder, 'confusion_mat'), cm)
        #     # plot confusion matrix
        #     plot_confusion_matrix(cm, self.opt.dataset, self.opt.save_folder)

        
        return acc, purity

