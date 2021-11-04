from utils.constants import *
from utils.generic import *
from tqdm import tqdm
import pickle
import os


class RecDataset:
    @staticmethod
    def parse_dp_args(parser):
        """
        DataProcessor related argument parser
        :param parser: argument parser
        :return: updated argument parser
        """
        parser.add_argument('--train_num_neg', type=int, default=1,
                            help='Negative sample num for each instance in train set.')
        parser.add_argument('--vt_num_neg', type=int, default=-1,
                            help='Number of negative sample in validation/testing stage.')
        return parser

    def __init__(self, data_reader, stage, batch_size=128, num_neg=1):
        self.data_reader = data_reader
        self.num_user = len(data_reader.user_ids_set)
        self.num_item = len(data_reader.item_ids_set)
        self.batch_size = batch_size
        self.stage = stage
        self.num_neg = num_neg
        # prepare test/validation dataset
        valid_pkl_path = os.path.join(self.data_reader.path, self.data_reader.dataset_name + VALID_PKL_SUFFIX)
        test_pkl_path = os.path.join(self.data_reader.path, self.data_reader.dataset_name + TEST_PKL_SUFFIX)
        if self.stage == 'valid':
            if os.path.exists(valid_pkl_path):
                with open(valid_pkl_path, 'rb') as file:
                    logging.info('Load validation data from pickle file.')
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(valid_pkl_path, 'wb') as file:
                    pickle.dump(self.data, file)
        elif self.stage == 'test':
            if os.path.exists(test_pkl_path):
                with open(test_pkl_path, 'rb') as file:
                    logging.info('Load test data from pickle file.')
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(test_pkl_path, 'wb') as file:
                    pickle.dump(self.data, file)
        else:
            self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _get_data(self):
        if self.stage == 'train':
            return self._get_train_data()
        else:
            return self._get_vt_data()

    def _get_train_data(self):
        df = self.data_reader.train_df
        df[SAMPLE_ID] = df.index
        columns_order = [USER, ITEM, SAMPLE_ID, LABEL] + [f_col for f_col in self.data_reader.feature_columns]
        data = df[columns_order].to_numpy()
        return data

    def _get_vt_data(self):
        if self.stage == 'valid':
            df = self.data_reader.validation_df
            logging.info('Prepare validation data...')
        elif self.stage == 'test':
            df = self.data_reader.test_df
            logging.info('Prepare test data...')
        else:
            raise ValueError('Wrong stage in dataset.')
        df[SAMPLE_ID] = df.index
        columns_order = [USER, ITEM, SAMPLE_ID, LABEL] + [f_col for f_col in self.data_reader.feature_columns]
        data = df[columns_order].to_numpy()

        total_batches = int((len(df) + self.batch_size - 1) / self.batch_size)
        batches = []
        for n_batch in tqdm(range(total_batches), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batch_start = n_batch * self.batch_size
            batch_end = min(len(df), batch_start + self.batch_size)

            real_batch_size = batch_end - batch_start

            batch = data[batch_start:batch_start + real_batch_size, :]

            inputs = np.asarray(batch)[:, 0:3]
            labels = np.asarray(batch)[:, 3]
            features = np.asarray(batch)[:, 4:]
            inputs = np.concatenate((inputs, features), axis=1)

            neg_samples = self._neg_samples_from_all(inputs, self.num_neg)
            neg_labels = np.asarray([0] * neg_samples.shape[0])

            tmp_sample = np.concatenate((inputs, neg_samples), axis=0)
            samples = torch.from_numpy(tmp_sample[:, 0:3])
            labels = torch.from_numpy(np.concatenate((labels, neg_labels), axis=0))
            features = torch.from_numpy(tmp_sample[:, 3:])

            feed_dict = {'X': samples, LABEL: labels, 'features': features}
            batches.append(feed_dict)
        return batches

    def collate_fn(self, batch):
        if self.stage == 'train':
            feed_dict = self._collate_train(batch)
        else:
            feed_dict = self._collate_vt(batch)
        return feed_dict

    def _collate_train(self, batch):
        inputs = np.asarray(batch)[:, 0:3]
        labels = np.asarray(batch)[:, 3]
        features = np.asarray(batch)[:, 4:]
        neg_samples = self._neg_sampler(inputs)
        neg_samples = np.insert(neg_samples, 0, inputs[:, 0], axis=1)
        neg_samples = np.insert(neg_samples, 2, inputs[:, 2], axis=1)
        neg_labels = np.asarray([0] * neg_samples.shape[0])
        neg_features = np.copy(features)
        assert len(inputs) == len(neg_samples)
        samples = torch.from_numpy(np.concatenate((inputs, neg_samples), axis=0))
        labels = torch.from_numpy(np.concatenate((labels, neg_labels), axis=0))
        features = torch.from_numpy((np.concatenate((features, neg_features), axis=0)))
        feed_dict = {'X': samples, LABEL: labels, 'features': features}
        return feed_dict

    @staticmethod
    def _collate_vt(data):
        return data

    def _neg_sampler(self, batch):
        neg_items = np.random.randint(1, self.num_item, size=(len(batch), self.num_neg))
        for i, (user, _, _) in enumerate(batch):
            user_clicked_set = self.data_reader.all_user2items_dict[user]
            for j in range(self.num_neg):
                while neg_items[i][j] in user_clicked_set:
                    neg_items[i][j] = np.random.randint(1, self.num_item)
        return neg_items

    def _neg_samples_from_all(self, batch, num_neg=-1):
        neg_items = None
        for idx, data in enumerate(batch):
            user = data[0]
            sample_id = data[2]
            features = data[3:]
            neg_candidates = list(self.data_reader.item_ids_set - self.data_reader.all_user2items_dict[user])
            if num_neg != -1:
                if num_neg <= len(neg_candidates):
                    neg_candidates = np.random.choice(neg_candidates, num_neg, replace=False)
                else:
                    neg_candidates = np.random.choice(neg_candidates, len(neg_candidates), replace=False)
            user_arr = np.asarray([user] * len(neg_candidates))
            id_arr = np.asarray([sample_id] * len(neg_candidates))
            feature_arr = np.tile(features, (len(neg_candidates), 1))
            neg_candidates = np.expand_dims(np.asarray(neg_candidates), axis=1)
            neg_candidates = np.insert(neg_candidates, 0, user_arr, axis=1)
            neg_candidates = np.insert(neg_candidates, 2, id_arr, axis=1)
            neg_candidates = np.concatenate((neg_candidates, feature_arr), axis=1)

            if neg_items is None:
                neg_items = neg_candidates
            else:
                neg_items = np.concatenate((neg_items, neg_candidates), axis=0)

        return neg_items


class DiscriminatorDataset:
    @staticmethod
    def parse_dp_args(parser):
        """
        DataProcessor related argument parser
        :param parser: argument parser
        :return: updated argument parser
        """
        parser.add_argument('--disc_batch_size', type=int, default=7000,
                            help='discriminator train batch size')
        return parser

    def __init__(self, data_reader, stage, batch_size=1000):
        self.data_reader = data_reader
        self.stage = stage
        self.batch_size = batch_size
        self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _get_data(self):
        if self.stage == 'train':
            return self._get_train_data()
        else:
            return self._get_test_data()

    def _get_train_data(self):
        data = self.data_reader.train_df.to_numpy()
        return data

    def _get_test_data(self):
        data = self.data_reader.test_df.to_numpy()
        return data

    @staticmethod
    def collate_fn(data):
        feed_dict = dict()
        feed_dict['X'] = torch.from_numpy(np.asarray(data)[:, 0])
        feed_dict['features'] = torch.from_numpy(np.asarray(data)[:, 1:])
        return feed_dict
