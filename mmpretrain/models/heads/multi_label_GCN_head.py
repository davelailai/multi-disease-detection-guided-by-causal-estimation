from typing import Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from torch.nn import Parameter
import math
import os

from mmpretrain.registry import MODELS
from .multi_label_cls_head import MultiLabelClsHead

'''Attention-Driven Dynamic Graph Convolutional Network for Multi-label Image Recognition'''

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

@MODELS.register_module()
class GCNClsHead_base(MultiLabelClsHead):
    def __init__(self,
                 num_classes: int,
                 label_file,
                 name,
                 t: int,
                 in_channels: int,
                 out_channels: int,
                 itself=True,
                 inter_channels=1024,
                 **kwargs):

        super(GCNClsHead_base, self).__init__( **kwargs)
        
        # self.num_classes = num_classes


        self.gc1 = GraphConvolution(in_channels, inter_channels)
        self.gc2 = GraphConvolution(inter_channels, out_channels)
        if label_file.endswith('pkl'):
            import pickle
            with open(label_file, 'rb') as file:
                A=pickle.load(file)
            A = A * 0.25 / (A.sum(0, keepdims=True) + 1e-6)   
            A = torch.from_numpy(A).float()
            A =self.gen_adj(A) 
            self.A=A

        else:
            self.A = self.gen_A(label_file, t, name, itself)

        # self.A = Parameter(torch.from_numpy(_adj).float())

        # self.adj=self.gen_adj(self.A).detach()

        query_embed = nn.Embedding(num_classes, in_channels)
        self.query_embed=query_embed.weight
        self.relu = nn.LeakyReLU(0.2)
        self.pooling = nn.MaxPool2d(14, 14)
        # self.adj= self.adj.to(self.query_embed.device)
    

        
    def forward(self, feature):
        # inp = inp[0]
        feature = self.pre_logits(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        # adj = self.gen_adj(self.A).detach()
        # self.adj= self.adj.to(self.query_embed.device)
        self.A=self.A.to(self.query_embed.device)
        x = self.gc1(self.query_embed.detach(), self.A)
        x = self.relu(x)
        x = self.gc2(x, self.A)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x
       

    def gen_A(self, file, t, name, itself=True): 
         
        co_occurrence_matrix=self.co_occurrence(file,name)
        if itself:
            pass
        else:
            import numpy as np
            np.fill_diagonal(co_occurrence_matrix, 0)
        _adj=co_occurrence_matrix/co_occurrence_matrix.sum(0)
        _adj[_adj < t] = 0
        # _adj[_adj >= t] = 1
        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)   
        _adj = torch.from_numpy(_adj).float()
        _adj =self.gen_adj(_adj) 
        # _adj[_adj < phi] = 0
        # _adj[_adj >= t] = 1

        # _adj = result['adj']
        # _nums = result['nums']
        # _nums = _nums[:, np.newaxis]
        # _adj = _adj / _nums
        # _adj[_adj < t] = 0
        # _adj[_adj >= t] = 1
        # _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        # _adj = _adj + np.identity(num_classes, np.int)
        return _adj

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj
    def co_occurrence(self, file, name):
        import pandas as pd
        from sklearn.preprocessing import MultiLabelBinarizer     
        try:
            all_labels = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                all_labels = pd.read_csv(file, encoding='latin1')
            except UnicodeDecodeError:
                try:
                    all_labels = pd.read_csv(file, encoding='utf-16')
                except UnicodeDecodeError:
                    print("Unable to decode the file using available encodings.")
                       
        if name=='ChestX14':
            pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule',
                    'Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema',
                    'Consolidation','Infiltration','Fibrosis','Pneumonia']
            index=['A','B','C','D','E','F','G','H','I','J','Q','L','M','N']
        if name=='FFA':
            pathology_list = ['Lesion_Leakage','Lesion_Accumulate','Lesion_Dyeing',
                # 'Lesion_Transparent,'
                'Lesion_Shading','Lesion_Inperfusion','Lesion_Vessel_Abnormal']
            index=['A','B','C','D','E','F']
        if name=='OIA-ODIR':
            pathology_list = ['D', 'G', 'C' ,'A', 'H' ,'M', 'O']
            index=['A','B','C','D','E','F','G']

        

        labels_dataset = pd.DataFrame(all_labels, columns=pathology_list)
        labels_dataset.columns=index
        labels_dataset['labels'] = labels_dataset.apply(self.merge_columns, axis=1)
        # labels_dataset['labels'] = labels_dataset.apply(lambda row: ', '.join(map(str, row)), axis=1)
        mlb = MultiLabelBinarizer()
        label_matrix = mlb.fit_transform(labels_dataset['labels'])
        label_df = pd.DataFrame(label_matrix, columns=mlb.classes_)
        co_occurrence_matrix = label_df.T.to_numpy() @ label_df.to_numpy()
        co_occurrence_matrix = co_occurrence_matrix[2:,2:]
        return co_occurrence_matrix
    def merge_columns(self, row):
        return ', '.join(col for col, value in row.items() if value)
    
   

        

    