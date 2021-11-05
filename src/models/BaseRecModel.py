# coding=utf-8
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import os
import numpy as np
from utils.constants import *


class BaseRecModel(nn.Module):
    """
    Base recommendation model. Child classes need to override:
    parse_model_args,
    __init__,
    _init_weights,
    predict,
    forward,
    """

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        """
        command line arguments
        :param parser: parser obj
        :param model_name: model name
        :return:
        """
        parser.add_argument('--model_path', type=str,
                            default='../model/%s/%s.pt' % (model_name, model_name),
                            help='Model save path.')
        parser.add_argument('--u_vector_size', type=int,
                            default=64,
                            help='user vector size')
        parser.add_argument('--i_vector_size', type=int,
                            default=64,
                            help='item vector size')
        parser.add_argument('--filter_mode', type=str,
                            default='combine',
                            help='combine for using one filter per sensitive feature,'
                                 'separate for using one filter per sensitive feature combination')
        return parser

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt', filter_mode='combine'):
        """
        :param data_processor_dict:
        :param user_num:
        :param item_num:
        :param u_vector_size:
        :param i_vector_size:
        :param random_seed:
        :param dropout:
        :param model_path:
        :param filter_mode: 'combine'-> for each combination train one filter;
        'separate' -> one filter for one sensitive feature, do combination for complex case.
        """
        super(BaseRecModel, self).__init__()
        self.data_processor_dict = data_processor_dict
        self.user_num = user_num
        self.item_num = item_num
        self.u_vector_size = u_vector_size
        self.i_vector_size = i_vector_size
        self.dropout = dropout
        self.random_seed = random_seed
        self.filter_mode = filter_mode
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.model_path = model_path

        self._init_nn()
        self._init_sensitive_filter()
        logging.debug(list(self.parameters()))

        self.total_parameters = self.count_variables()
        logging.info('# of params: %d' % self.total_parameters)

        # optimizer assigned by *_runner.py
        self.optimizer = None

    def _init_nn(self):
        """
        Initialize neural networks
        :return:
        """
        pass

    def _init_sensitive_filter(self):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim)
            )
            return sequential
        num_features = len(self.data_processor_dict['train'].data_reader.feature_columns)
        self.filter_num = num_features if self.filter_mode == 'combine' else 2**num_features
        self.num_features = num_features
        self.filter_dict = nn.ModuleDict(
            {str(i + 1): get_sensitive_filter(self.u_vector_size) for i in range(self.filter_num)})

    def apply_filter(self, vectors, filter_mask):
        if self.filter_mode == 'separate' and np.sum(filter_mask) != 0:
            filter_mask = np.asarray(filter_mask)
            idx = filter_mask.dot(2**np.arange(filter_mask.size))
            sens_filter = self.filter_dict[str(idx)]
            result = sens_filter(vectors)
        elif self.filter_mode == 'combine' and np.sum(filter_mask) != 0:
            result = None
            for idx, val in enumerate(filter_mask):
                if val != 0:
                    sens_filter = self.filter_dict[str(idx + 1)]
                    result = sens_filter(vectors) if result is None else result + sens_filter(vectors)
            result = result / np.sum(filter_mask)   # average the embedding
        else:
            result = vectors
        return result

    def count_variables(self):
        """
        Total number of parameters in the model
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def l2(self):
        """
        calc the summation of l2 of all parameters
        :return:
        """
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return l2

    def predict(self, feed_dict, filter_mask):
        """
        prediction only without loss calculation
        :param feed_dict: input dictionary
        :param filter_mask: mask for filter selection
        :return: output dictionary，with keys (at least)
                "prediction": predicted values;
                "check": intermediate results to be checked and printed out
        """
        check_list = []
        x = self.x_bn(feed_dict['X'].float())
        x = torch.nn.Dropout(p=feed_dict['dropout'])(x)
        prediction = F.relu(self.prediction(x)).view([-1])
        out_dict = {'prediction': prediction,
                    'check': check_list}
        return out_dict

    def forward(self, feed_dict, filter_mask):
        out_dict = self.predict(feed_dict, filter_mask)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        return out_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)

    def freeze_model(self):
        self.eval()
        for params in self.parameters():
            params.requires_grad = False
