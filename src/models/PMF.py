import torch
from models.BaseRecModel import BaseRecModel


class PMF(BaseRecModel):
    def _init_nn(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.u_vector_size)

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        pmf_u_vectors = self.apply_filter(pmf_u_vectors, filter_mask)

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
        return out_dict
