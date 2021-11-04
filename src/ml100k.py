# coding=utf-8
import os
import pandas as pd
from collections import defaultdict, Counter
from utils.generic import *
from utils.constants import *

np.random.seed(2018)

RAW_DATA = '../data/'
RATINGS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.data')
# RATINGS_FILE = os.path.join(RAW_DATA, 'yelp.csv')
# RATINGS = pd.read_csv(RATINGS_FILE, sep='\t', header=None)
USERS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.user')
USERS = pd.read_csv(USERS_FILE, sep='|', header=None)
ITEMS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.item')
# ITEMS = pd.read_csv(ITEMS_FILE, sep='|', header=None)

USER_FEATURE_FILE = os.path.join(DATA_DIR, 'ml100k.users.csv')
# ITEM_FEATURE_FILE = os.path.join(global_p.DATA_DIR, 'ml100k.items.csv')

ALL_DATA_FILE = os.path.join(DATA_DIR, 'ml100k.all.csv')
# ALL_DATA_FILE = os.path.join(global_p.DATA_DIR, 'ml100k01.all.csv')
# ALL_DATA_FILE = os.path.join(global_p.DATA_DIR, 'yelp01.all.csv')


def format_user_feature(out_file):
    print('format_user_feature', USERS_FILE)
    user_df = pd.read_csv(USERS_FILE, sep='|', header=None)
    user_df = user_df[[0, 1, 2, 3]]
    user_df.columns = ['uid', 'u_age', 'u_gender', 'u_occupation']
    min_age, max_age = 10, 60
    user_df['u_age'] = user_df['u_age'].apply(
        lambda x: 0 if x < min_age else int(x / 5) - 1 if x <= max_age else int(max_age / 5) if x > max_age else 0)
    user_df['u_gender'] = user_df['u_gender'].apply(lambda x: defaultdict(int, {'M': 0, 'F': 1})[x])
    occupation = {'none': 0, 'other': 1}
    for o in user_df['u_occupation'].unique():
        if o not in occupation:
            occupation[o] = len(occupation)
    user_df['u_occupation'] = user_df['u_occupation'].apply(lambda x: defaultdict(int, occupation)[x])
    # print(Counter(user_df['u_occupation']))
    # print(user_df)
    # user_df.info(null_counts=True)
    # print(user_df.min())
    user_df.to_csv(out_file, index=False, sep='\t')
    return user_df


def format_item_feature(out_file):
    print('format_item_feature', ITEMS_FILE, out_file)
    item_df = pd.read_csv(ITEMS_FILE, sep='|', header=None, encoding="ISO-8859-1")
    item_df = item_df.drop([1, 3, 4], axis=1)
    item_df.columns = ['iid', 'i_year',
                       'i_Action', 'i_Adventure', 'i_Animation', "i_Children's", 'i_Comedy',
                       'i_Crime', 'i_Documentary ', 'i_Drama ', 'i_Fantasy ', 'i_Film-Noir ',
                       'i_Horror ', 'i_Musical ', 'i_Mystery ', 'i_Romance ', 'i_Sci-Fi ',
                       'i_Thriller ', 'i_War ', 'i_Western', 'i_Other']
    item_df['i_year'] = item_df['i_year'].apply(lambda x: int(str(x).split('-')[-1]) if pd.notnull(x) else -1)
    seps = [0, 1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_year'].max() + 2)))
    year_dict = {}
    for i, sep in enumerate(seps[:-1]):
        for j in range(seps[i], seps[i + 1]):
            year_dict[j] = i + 1
    item_df['i_year'] = item_df['i_year'].apply(lambda x: defaultdict(int, year_dict)[x])
    for c in item_df.columns[2:]:
        item_df[c] = item_df[c] + 1
    # print(Counter(item_df['i_year']))
    # print(item_df)
    # item_df.info(null_counts=True)
    item_df.to_csv(out_file, index=False, sep='\t')
    return item_df


def format_all_inter(out_file, mapping_path, label01=False):
    print('format_all_inter', RATINGS_FILE, out_file)
    inter_df = pd.read_csv(RATINGS_FILE, sep='\t', header=None)
    inter_df.columns = ['uid', 'iid', 'label', 'time']
    inter_df = inter_df.sort_values(by='time')
    inter_df = inter_df.drop(columns=['time'])
    inter_df = inter_df.drop_duplicates(['uid', 'iid']).reset_index(drop=True)

    user_mapping_file = os.path.join(mapping_path, 'user_mapping.tsv')
    item_mapping_file = os.path.join(mapping_path, 'item_mapping.tsv')

    # 给uid编号，从1开始
    uids = sorted(inter_df['uid'].unique())
    uid_dict = dict(zip(uids, range(1, len(uids) + 1)))
    mapping_df = pd.DataFrame.from_dict(uid_dict, orient='index', columns=['mapped'])
    mapping_df.index.name = 'original'
    mapping_df = mapping_df.reset_index()
    mapping_df.to_csv(user_mapping_file, sep='\t', index=False)
    inter_df['uid'] = inter_df['uid'].apply(lambda x: uid_dict[x])

    # 给iid编号，从1开始
    iids = sorted(inter_df['iid'].unique())
    iid_dict = dict(zip(iids, range(1, len(iids) + 1)))
    mapping_df = pd.DataFrame.from_dict(iid_dict, orient='index', columns=['mapped'])
    mapping_df.index.name = 'original'
    mapping_df = mapping_df.reset_index()
    mapping_df.to_csv(item_mapping_file, sep='\t', index=False)
    inter_df['iid'] = inter_df['iid'].apply(lambda x: iid_dict[x])

    user_df = pd.read_csv(USER_FEATURE_FILE, sep='\t')
    # item_df = pd.read_csv(ITEM_FEATURE_FILE, sep='\t')
    inter_df = pd.merge(inter_df, user_df, on='uid', how='left')
    # inter_df = pd.merge(inter_df, item_df, on='iid', how='left')
    # print(inter_df)
    # inter_df.info(null_counts=True)
    # inter_df['label'] -= inter_df['label'].min()
    # print(inter_df)
    if label01:
        inter_df['label'] = inter_df['label'].apply(lambda x: 1 if x > 0 else 0)
    print('label:', inter_df['label'].min(), inter_df['label'].max())
    print(Counter(inter_df['label']))
    # inter_df.columns = ['head', 'tail', 'label', 'relation']
    # inter_df = inter_df[['head', 'tail', 'relation', 'label']]
    inter_df.to_csv(out_file, sep='\t', index=False)
    return inter_df


def random_split_data(all_data_file, dataset_name, vt_ratio=0.1):
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('random_split_data', dir_name)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    all_data.to_csv(os.path.join(dir_name, dataset_name + '.all.tsv'), index=False, sep='\t')
    vt_size = int(len(all_data) * vt_ratio)
    validation_set = all_data.sample(n=vt_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=vt_size).sort_index()
    train_set = all_data.drop(test_set.index)
    # write files
    train_set.to_csv(os.path.join(dir_name, dataset_name + '.train.tsv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, dataset_name + '.validation.tsv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, dataset_name + '.test.tsv'), index=False, sep='\t')


def main():
    format_user_feature(USER_FEATURE_FILE)
    # format_item_feature(ITEM_FEATURE_FILE)
    dataset_name = 'ml100k-1-5'
    mapping_path = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(mapping_path):
        os.mkdir(mapping_path)
    format_all_inter(ALL_DATA_FILE, mapping_path=mapping_path, label01=True)
    # format_all_inter(ALL_DATA_FILE, label01=True)

    # random_split_data(ALL_DATA_FILE, dataset_name, u_f=USER_FEATURE_FILE, i_f=ITEM_FEATURE_FILE)
    random_split_data(ALL_DATA_FILE, dataset_name)

    # dataset_name = 'ml100k01-1-5'
    # dataset_name = 'yelp-1-5'
    # leave_out_by_time(ALL_DATA_FILE, dataset_name, leave_n=1, warm_n=5,
    #                   u_f=None, i_f=None)
    return


if __name__ == '__main__':
    main()
