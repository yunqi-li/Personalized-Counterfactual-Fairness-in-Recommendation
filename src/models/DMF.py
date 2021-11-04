# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRecModel import BaseRecModel


class DMF(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='DMF'):
        parser.add_argument('--num_layers', type=int, default=3,
                            help="Number of mlp layers.")
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 num_layers=3, random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt',
                 filter_mode='combine'):
        self.num_layers = num_layers
        BaseRecModel.__init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                              random_seed=random_seed, dropout=dropout, model_path=model_path,
                              filter_mode=filter_mode)

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='sigmoid')
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _init_nn(self):
        self.uid_embeddings = nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.u_vector_size)

        self.cos = nn.CosineSimilarity()

        self.u_mlp = nn.ModuleList([nn.Linear(self.u_vector_size, self.u_vector_size)])
        # self.u_mlp = nn.ModuleList([nn.Linear(self.item_num, self.ui_vector_size)])
        for layer in range(self.num_layers - 1):
            self.u_mlp.append(nn.Linear(self.u_vector_size, self.u_vector_size))
        self.i_mlp = nn.ModuleList([nn.Linear(self.u_vector_size, self.u_vector_size)])
        # self.i_mlp = nn.ModuleList([nn.Linear(self.user_num, self.ui_vector_size)])
        for layer in range(self.num_layers - 1):
            self.i_mlp.append(nn.Linear(self.u_vector_size, self.u_vector_size))

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        user_embeddings = self.uid_embeddings(u_ids)
        item_embeddings = self.iid_embeddings(i_ids)
        u_input = user_embeddings

        for layer in self.u_mlp:
            u_input = layer(u_input)
            u_input = F.relu(u_input)
            u_input = torch.nn.Dropout(p=self.dropout)(u_input)

        i_input = item_embeddings
        for layer in self.i_mlp:
            i_input = layer(i_input)
            i_input = F.relu(i_input)
            i_input = torch.nn.Dropout(p=self.dropout)(i_input)

        # prediction = F.relu(self.cos(u_input, i_input)).view([-1]) * 10
        prediction = self.cos(u_input, i_input).view([-1]) * 10
        # check_list.append(('prediction', prediction))
        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': u_input}
        return out_dict
