# coding=utf-8

import torch
from models.BaseRecModel import BaseRecModel


class BiasedMF(BaseRecModel):
    def _init_nn(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.i_vector_size)
        assert self.u_vector_size == self.i_vector_size
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)

        cf_u_vectors = self.apply_filter(cf_u_vectors, filter_mask)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': cf_u_vectors}
        return out_dict
