from utils.utils import *

ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'elementary': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'obqa': {
        'lstm': 3e-4,
        'openai-gpt': 3e-5,
        'bert-base-cased': 1e-4,
        'bert-large-cased': 1e-4,
        'roberta-large': 1e-5,
    }
}

DATASET_LIST = ['csqa', 'obqa', 'socialiqa', "elementary"]

DATASET_SETTING = {
    'csqa': 'inhouse',
    'obqa': 'official',
    'socialiqa': 'official',
    "elementary": "official"
}

DATASET_NO_TEST = ['socialiqa']

EMB_PATHS = {
    'transe': './data/transe/glove.transe.sgd.ent.npy',
    'lm': './data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': './data/transe/concept.nb.npy',
    'tzw': './data/cpnet/tzw.ent.npy',
}


def add_data_arguments(parser):
    # arguments that all datasets share
    parser.add_argument('--ent_emb', default=['tzw'], choices=['transe', 'numberbatch', 'lm', 'tzw'], nargs='+', help='sources for entity embeddings')
    parser.add_argument('--ent_emb_paths', default=['./data/transe/glove.transe.sgd.ent.npy'], nargs='+', help='paths to entity embedding file(s)')
    parser.add_argument('--rel_emb_path', default='./data/transe/glove.transe.sgd.rel.npy', help='paths to relation embedding file')
    # dataset specific
    parser.add_argument('-ds', '--dataset', default='csqa', choices=DATASET_LIST, help='dataset name')
    parser.add_argument('-ih', '--inhouse', default=True, type=bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='./data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')
    # statements
    parser.add_argument('--train_statements', default='./data/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--dev_statements', default='./data/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--test_statements', default='./data/{dataset}/statement/test.statement.jsonl')
    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=64, type=int)
    parser.add_argument('--format', default=[], choices=['add_qa_prefix', 'no_extra_sep', 'fairseq', 'add_prefix_space'], nargs='*')
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb],
                        inhouse=(DATASET_SETTING[args.dataset] == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset))
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    # used only for LSTM encoder
    parser.add_argument('--encoder_dim', default=128, type=int, help='number of LSTM hidden units')
    parser.add_argument('--encoder_layer_num', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--encoder_bidir', default=True, type=bool_flag, nargs='?', const=True, help='use BiLSTM')
    parser.add_argument('--encoder_dropoute', default=0.1, type=float, help='word dropout')
    parser.add_argument('--encoder_dropouti', default=0.1, type=float, help='dropout applied to embeddings')
    parser.add_argument('--encoder_dropouth', default=0.1, type=float, help='dropout applied to lstm hidden states')
    parser.add_argument('--encoder_pretrained_emb', default='./data/glove/glove.6B.300d.npy', help='path to pretrained emb in .npy format')
    parser.add_argument('--encoder_freeze_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze lstm input embedding layer')
    parser.add_argument('--encoder_pooler', default='max', choices=['max', 'mean'], help='pooling function')
    args, _ = parser.parse_known_args()
    parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=2, type=int, help='stop training if dev does not increase for N epochs')


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    parser.add_argument('--kb', type=str, default="conceptnet", help='Name of the knowledge base')
    parser.add_argument('--min_path_length', type=int, default=2, help="The minimum length of a path")
    parser.add_argument('--max_path_length', type=int, default=5, help="The maximum length of a path")
    parser.add_argument('--max_num_paths', type=int, default=100, help="The maximum number of paths to consider")
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)


def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser


def get_lstm_config_from_args(args):
    lstm_config = {
        'hidden_size': args.encoder_dim,
        'output_size': args.encoder_dim,
        'num_layers': args.encoder_layer_num,
        'bidirectional': args.encoder_bidir,
        'emb_p': args.encoder_dropoute,
        'input_p': args.encoder_dropouti,
        'hidden_p': args.encoder_dropouth,
        'pretrained_emb_or_path': args.encoder_pretrained_emb,
        'freeze_emb': args.encoder_freeze_emb,
        'pool_function': args.encoder_pooler,
    }
    return lstm_config
