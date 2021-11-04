# coding=utf-8
import sys
import os
from utils.generic import *
from data_reader import RecDataReader
from runner import RecRunner
from datasets import RecDataset
from models.BiasedMF import BiasedMF
from models.PMF import PMF
from models.MLP import MLP
from models.DMF import DMF
from models.BaseRecModel import BaseRecModel
from models.Discriminators import Discriminator, BinaryDiscriminator, BinaryAttacker, MultiClassAttacker
from data_reader import DiscriminatorDataReader
from datasets import DiscriminatorDataset
from torch.utils.data import DataLoader


def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--data_reader', type=str, default='RecDataReader',
                             help='Choose data_reader')
    init_parser.add_argument('--data_processor', type=str, default='RecDataset',
                             help='Choose data_processor')
    init_parser.add_argument('--model_name', type=str, default='BiasedMF',
                             help='Choose model to run.')
    init_parser.add_argument('--runner', type=str, default='RecRunner',
                             help='Choose runner')
    init_args, init_extras = init_parser.parse_known_args()

    # choose data_reader
    data_reader_name = eval(init_args.data_reader)

    # choose model
    model_name = eval(init_args.model_name)
    runner_name = eval(init_args.runner)

    # choose data_processor
    data_processor_name = eval(init_args.data_processor)

    # cmd line paras
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = data_reader_name.parse_data_args(parser)
    parser = Discriminator.parse_disc_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)
    parser = DiscriminatorDataset.parse_dp_args(parser)

    args, extras = parser.parse_known_args()

    # log,model,result filename
    log_file_name = [init_args.model_name,
                     args.dataset, str(args.random_seed),
                     'optimizer=' + args.optimizer, 'lr=' + str(args.lr), 'l2=' + str(args.l2),
                     'dropout=' + str(args.dropout), 'batch_size=' + str(args.batch_size)]
    log_file_name = '__'.join(log_file_name).replace(' ', '__')
    if args.log_file == '../log/log.txt':
        args.log_file = '../log/%s.txt' % log_file_name
    if args.result_file == '../result/result.npy':
        args.result_file = '../result/%s.npy' % log_file_name
    if args.model_path == '../model/%s/%s.pt' % (init_args.model_name, init_args.model_name):
        args.model_path = '../model/%s/%s.pt' % (init_args.model_name, log_file_name)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # convert the namespace into dictionary e.g. init_args.model_name -> {'model_name': BaseModel}
    logging.info(vars(init_args))
    logging.info(vars(args))

    logging.info('DataReader: ' + init_args.data_reader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner)
    logging.info('DataProcessor: ' + init_args.data_processor)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info("# cuda devices: %d" % torch.cuda.device_count())
    # create data_reader
    data_reader = data_reader_name(path=args.path, dataset_name=args.dataset, sep=args.sep)

    # create data processor
    data_processor_dict = {}
    for stage in ['train', 'valid', 'test']:
        if stage == 'train':
            if init_args.data_processor in ['RecDataset']:
                data_processor_dict[stage] = data_processor_name(
                    data_reader, stage, batch_size=args.batch_size, num_neg=args.train_num_neg)
            else:
                raise ValueError('Unknown DataProcessor')
        else:
            if init_args.data_processor in ['RecDataset']:
                data_processor_dict[stage] = data_processor_name(
                    data_reader, stage, batch_size=args.vt_batch_size, num_neg=args.vt_num_neg)
            else:
                raise ValueError('Unknown DataProcessor')

    # create model
    if init_args.model_name in ['BiasedMF', 'PMF']:
        model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                           item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                           i_vector_size=args.i_vector_size, random_seed=args.random_seed, dropout=args.dropout,
                           model_path=args.model_path, filter_mode=args.filter_mode)
    elif init_args.model_name in ['DMF', 'MLP']:
        model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                           item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                           i_vector_size=args.i_vector_size, num_layers=args.num_layers,
                           random_seed=args.random_seed, dropout=args.dropout,
                           model_path=args.model_path, filter_mode=args.filter_mode)
    else:
        logging.error('Unknown Model: ' + init_args.model_name)
        return
    # init model paras
    model.apply(model.init_weights)

    # use gpu
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    # create discriminators
    fair_disc_dict = {}
    for feat_idx in data_reader.feature_info:
        fair_disc_dict[feat_idx] = \
            Discriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                          random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                          model_dir_path=os.path.dirname(args.model_path))
        fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)
        if torch.cuda.device_count() > 0:
            fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()
    # for feat_idx in data_reader.feature_info:
    #     if data_reader.feature_info[feat_idx].num_class == 2:
    #         fair_disc_dict[feat_idx] = \
    #             BinaryDiscriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
    #                                 random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
    #                                 model_dir_path=os.path.dirname(args.model_path))
    #     else:
    #         fair_disc_dict[feat_idx] = \
    #             Discriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
    #                           random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
    #                           model_dir_path=os.path.dirname(args.model_path))
    #     fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)
    #     if torch.cuda.device_count() > 0:
    #         fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()

    # create runner
    # batch_size is the training batch size, eval_batch_size is the batch size for evaluation
    if init_args.runner in ['BaseRunner']:
        runner = runner_name(
            optimizer=args.optimizer, learning_rate=args.lr,
            epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
            dropout=args.dropout, l2=args.l2,
            metrics=args.metric, check_epoch=args.check_epoch, early_stop=args.early_stop)
    elif init_args.runner in ['RecRunner']:
        runner = runner_name(
            optimizer=args.optimizer, learning_rate=args.lr,
            epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
            dropout=args.dropout, l2=args.l2,
            metrics=args.metric, check_epoch=args.check_epoch, early_stop=args.early_stop, num_worker=args.num_worker,
            no_filter=args.no_filter, reg_weight=args.reg_weight, d_steps=args.d_steps, disc_epoch=args.disc_epoch)
    else:
        logging.error('Unknown Runner: ' + init_args.runner)
        return

    if args.load > 0:
        model.load_model()
        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()
    if args.train > 0:
        runner.train(model, data_processor_dict, fair_disc_dict, skip_eval=args.skip_eval, fix_one=args.fix_one)

    # reset seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    if args.eval_disc:
        # Train extra discriminator for evaluation
        # create data reader
        disc_data_reader = DiscriminatorDataReader(path=args.path, dataset_name=args.dataset, sep=args.sep)

        # create data processor
        extra_data_processor_dict = {}
        for stage in ['train', 'test']:
            extra_data_processor_dict[stage] = DiscriminatorDataset(disc_data_reader, stage, args.disc_batch_size)

        # create discriminators
        extra_fair_disc_dict = {}
        for feat_idx in disc_data_reader.feature_info:
            if disc_data_reader.feature_info[feat_idx].num_class == 2:
                extra_fair_disc_dict[feat_idx] = \
                    BinaryAttacker(args.u_vector_size, disc_data_reader.feature_info[feat_idx],
                                   random_seed=args.random_seed, dropout=args.dropout,
                                   neg_slope=args.neg_slope, model_dir_path=os.path.dirname(args.model_path),
                                   model_name='eval')
            else:
                extra_fair_disc_dict[feat_idx] = \
                    MultiClassAttacker(args.u_vector_size, disc_data_reader.feature_info[feat_idx],
                                       random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                                       model_dir_path=os.path.dirname(args.model_path), model_name='eval')
            extra_fair_disc_dict[feat_idx].apply(extra_fair_disc_dict[feat_idx].init_weights)
            if torch.cuda.device_count() > 0:
                extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()

        if args.load_attack:
            for idx in extra_fair_disc_dict:
                logging.info('load attacker model...')
                extra_fair_disc_dict[idx].load_model()
        model.load_model()
        model.freeze_model()
        runner.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict, args.lr_attack,
                                   args.l2_attack)

    test_data = DataLoader(data_processor_dict['test'], batch_size=None, num_workers=args.num_worker,
                           pin_memory=True, collate_fn=data_processor_dict['test'].collate_fn)

    test_result_dict = dict()
    if args.no_filter:
        test_result = runner.evaluate(model, test_data)
    else:
        test_result, test_result_dict = runner.eval_multi_combination(model, test_data, args.fix_one)

    if args.no_filter:
        logging.info("Test After Training = %s "
                     % (format_metric(test_result)) + ','.join(runner.metrics))
    else:
        logging.info("Test After Training:\t Average: %s "
                     % (format_metric(test_result)) + ','.join(runner.metrics))
        for key in test_result_dict:
            logging.info("test= %s "
                         % (format_metric(test_result_dict[key])) + ','.join(runner.metrics) +
                         ' (' + key + ') ')

    return


if __name__ == '__main__':
    main()
