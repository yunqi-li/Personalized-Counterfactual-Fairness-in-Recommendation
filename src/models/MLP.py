# coding=utf-8

import torch
import torch.nn as nn
from models.BaseRecModel import BaseRecModel


class MLP(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='MLP'):
        parser.add_argument('--num_layers', type=int, default=3,
                            help="Number of mlp layers.")
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 num_layers=3, random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt',
                 filter_mode='combine'):
        self.num_layers = num_layers
        self.factor_size = u_vector_size // (2 ** (self.num_layers - 1))
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
        # Init embeddings
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.u_vector_size)

        # Init MLP
        self.mlp = nn.ModuleList([])
        pre_size = self.factor_size * (2 ** self.num_layers)
        for i in range(self.num_layers):
            self.mlp.append(nn.Dropout(p=self.dropout))
            self.mlp.append(nn.Linear(pre_size, pre_size // 2))
            self.mlp.append(nn.ReLU())
            pre_size = pre_size // 2
        self.mlp = nn.Sequential(*self.mlp)

        # Init predictive layer
        self.p_layer = nn.ModuleList([])
        assert pre_size == self.factor_size
        # pre_size = pre_size * 2
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_i_vectors = self.iid_embeddings(i_ids)

        mlp_u_vectors = self.apply_filter(mlp_u_vectors, filter_mask)

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=-1)
        mlp = self.mlp(mlp)

        # output = torch.cat((gmf, mlp), dim=-1)

        # prediction = self.prediction(output).view([-1])
        prediction = self.prediction(mlp).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors}
        return out_dict
