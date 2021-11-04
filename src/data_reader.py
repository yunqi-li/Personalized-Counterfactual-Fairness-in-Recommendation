import numpy as np
import os
import logging
import pandas as pd
from collections import defaultdict, namedtuple, OrderedDict
from utils.constants import *


class DataReader:
    def __init__(self, path, dataset_name, sep='\t', seq_sep=','):
        self.path = os.path.join(path, dataset_name)
        self.dataset_name = dataset_name
        self.sep = sep
        self.seq_sep = seq_sep
        self.train_file = os.path.join(self.path, dataset_name + TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset_name + VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset_name + TEST_SUFFIX)
        self.all_file = os.path.join(self.path, dataset_name + ALL_SUFFIX)
        self.feature_file = os.path.join(self.path, dataset_name + FEATURE_SUFFIX)
        self._load_data()
        self.features = self._load_feature() if os.path.exists(self.feature_file) else None

    def _load_data(self):
        if os.path.exists(self.all_file):
            logging.info("load all csv...")
            self.all_df = pd.read_csv(self.all_file, sep=self.sep)
        else:
            raise FileNotFoundError('all file is not found.')
        if os.path.exists(self.train_file):
            logging.info("load train csv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            logging.info("size of train: %d" % len(self.train_df))
        else:
            raise FileNotFoundError('train file is not found.')
        if os.path.exists(self.validation_file):
            logging.info("load validation csv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logging.info("size of validation: %d" % len(self.validation_df))
        else:
            raise FileNotFoundError('validation file is not found.')
        if os.path.exists(self.test_file):
            logging.info("load test csv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            logging.info("size of test: %d" % len(self.test_df))
        else:
            raise FileNotFoundError('test file is not found.')

    def _load_feature(self):
        """
        load pre-trained/feature embeddings. It is saved as a numpy text file.
        :return:
        """
        return np.loadtxt(self.feature_file, dtype=np.float32)


class RecDataReader(DataReader):
    @staticmethod
    def parse_data_args(parser):
        """
        DataProcessor related argument parser
        :param parser: argument parser
        :return: updated argument parser
        """
        parser.add_argument('--path', type=str, default='../dataset/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml1M',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--label', type=str, default='label',
                            help='name of dataset label column.')
        return parser

    def __init__(self, path, dataset_name, sep='\t', seq_sep=','):
        super().__init__(path, dataset_name, sep, seq_sep)
        self.user_ids_set = set(self.all_df[USER].tolist())
        self.item_ids_set = set(self.all_df[ITEM].tolist())
        self.num_nodes = len(self.user_ids_set) + len(self.item_ids_set)
        self.train_item2users_dict = self._prepare_item2users_dict(self.train_df)

        self.all_user2items_dict = self._prepare_user2items_dict(self.all_df)
        self.train_user2items_dict = self._prepare_user2items_dict(self.train_df)
        self.valid_user2items_dict = self._prepare_user2items_dict(self.validation_df)
        self.test_user2items_dict = self._prepare_user2items_dict(self.test_df)
        # add feature info for discriminator and filters
        uid_iid_label = [USER, ITEM, LABEL]
        self.feature_columns = [name for name in self.train_df.columns.tolist() if name not in uid_iid_label]
        Feature = namedtuple('Feature', ['num_class', 'label_min', 'label_max', 'name'])
        self.feature_info = \
            OrderedDict({idx + 1: Feature(self.all_df[col].nunique(), self.all_df[col].min(), self.all_df[col].max(),
                                          col) for idx, col in enumerate(self.feature_columns)})
        self.num_features = len(self.feature_columns)

    @staticmethod
    def _prepare_user2items_dict(df):
        df_groups = df.groupby(USER)
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group[ITEM].tolist())
        return user_sample_dict

    @staticmethod
    def _prepare_item2users_dict(df):
        df_groups = df.groupby(ITEM)
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group[USER].tolist())
        return user_sample_dict


class DiscriminatorDataReader:
    def __init__(self, path, dataset_name, sep='\t', seq_sep=',', test_ratio=0.1):
        self.path = os.path.join(path, dataset_name)
        self.sep = sep
        self.seq_sep = seq_sep
        self.all_file = os.path.join(self.path, dataset_name + ALL_SUFFIX)
        self.train_attacker_file = os.path.join(self.path, dataset_name + '.attacker' + TRAIN_SUFFIX)
        self.test_attacker_file = os.path.join(self.path, dataset_name + '.attacker' + TEST_SUFFIX)
        self.all_df = pd.read_csv(self.all_file, sep='\t')

        # add feature info for discriminator and filters
        uid_iid_label = [USER, ITEM, LABEL]
        self.feature_columns = [name for name in self.all_df.columns.tolist() if name not in uid_iid_label]

        Feature = namedtuple('Feature', ['num_class', 'label_min', 'label_max', 'name'])
        self.feature_info = \
            OrderedDict({idx + 1: Feature(self.all_df[col].nunique(), self.all_df[col].min(), self.all_df[col].max(),
                                          col) for idx, col in enumerate(self.feature_columns)})
        self.f_name_2_idx = {f_name: idx + 1 for idx, f_name in enumerate(self.feature_columns)}
        self.num_features = len(self.feature_columns)
        if os.path.exists(self.train_attacker_file) and os.path.exists(self.test_attacker_file):
            self.train_df = pd.read_csv(self.train_attacker_file, sep='\t')
            self.test_df = pd.read_csv(self.test_attacker_file, sep='\t')
        else:
            self.train_df, self.test_df = self._init_feature_df(self.all_df, test_ratio)

    def _init_feature_df(self, all_df, test_ratio):
        logging.info('Initializing attacker train/test file...')
        feature_df = pd.DataFrame()
        all_df = all_df.sort_values(by='uid')
        all_group = all_df.groupby('uid')

        uid_list = []
        feature_list_dict = {key: [] for key in self.feature_columns}
        for uid, group in all_group:
            uid_list.append(uid)
            for key in feature_list_dict:
                feature_list_dict[key].append(group[key].tolist()[0])
        feature_df[USER] = uid_list
        for f in self.feature_columns:
            feature_df[f] = feature_list_dict[f]

        test_size = int(len(feature_df) * test_ratio)
        sign = True
        counter = 0
        while sign:
            test_set = feature_df.sample(n=test_size).sort_index()
            for f in self.feature_columns:
                num_class = self.feature_info[self.f_name_2_idx[f]].num_class
                val_range = set([i for i in range(num_class)])
                test_range = set(test_set[f].tolist())
                if len(val_range) != len(test_range):
                    sign = True
                    break
                else:
                    sign = False
            print(counter)
            counter += 1

        train_set = feature_df.drop(test_set.index)
        train_set.to_csv(self.train_attacker_file, sep='\t', index=False)
        test_set.to_csv(self.test_attacker_file, sep='\t', index=False)
        return train_set, test_set
