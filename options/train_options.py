# note: currently doesn't handle recursively nested groups and conflicting
# option strings

import argparse
import oyaml as yaml
import sys
import os
from collections import OrderedDict

class TrainOptions():
    def __init__(self):
        self.initialized = False
        self.parser = parser = argparse.ArgumentParser("Training Parser")

    def initialize(self):
        parser = self.parser
        parser.add_argument('--config_file', type=argparse.FileType(mode='r'), help="configuration yml file")
        self.parser.add_argument('--overwrite_config', action='store_true', help="overwrite config files if they exist")
        self.parser.add_argument('--model', default='biggan', help="pretrained model to use, e.g. biggan, stylegan")
        parser.add_argument('--transform', default="zoom", help="transform operation, e.g. zoom, shiftx, color, rotate2d"),
        parser.add_argument('--num_samples', type=int, default=20000, help='number of latent z samples')
        parser.add_argument('--loss', type=str, default='l2', help='loss to use for training', choices=['l2', 'lpips'])
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
        parser.add_argument('--walk_type', type=str, default='NNz', choices=['NNz', 'linear'], help='type of latent z walk')
        parser.add_argument('--models_dir', type=str, default="./models", help="output directory for saved checkpoints")
        parser.add_argument('--model_save_freq', type=int, default=400, help="saves checkpoints after this many batches")
        parser.add_argument('--name', type=str, help="experiment name, saved within models_dir")
        parser.add_argument('--suffix', type=str, help="suffix for experiment name")
        parser.add_argument('--prefix', type=str, help="prefix for experiment name")
        parser.add_argument("--gpu", default="", type=str, help='GPUs to use (leave blank for CPU only)')

        # NN walk parameters
        group = parser.add_argument_group('nn', 'parameters used to specify NN walk')
        group.add_argument('--eps', type=float, help="step size of each NN block")
        group.add_argument('--num_steps', type=int, help="number of NN blocks")

        # color transformation parameters
        group = parser.add_argument_group('color', 'parameters used for color walk')
        group.add_argument('--channel', type=int, help="which channel to modify; if unspecified, modifies all channels for linear walks, and luminance for NN walks")

        # biggan walk parameters
        group = parser.add_argument_group('biggan', 'parameters used for biggan walk')
        group.add_argument('--category', type=int, help="which category to train on; if unspecified uses all categories")

        # stylegan walk parameters
        group = parser.add_argument_group('stylegan', 'parameters used for stylegan walk')
        group.add_argument('--dataset', default="cars", help="which dataset to use for pretrained stylegan, e.g. cars, cats, celebahq")
        group.add_argument('--latent', default="w", help="which latent space to use; z or w")
        group.add_argument('--truncation_psi', default=1.0, help="truncation for NN walk in w")

        # pgan walk parameters
        group = parser.add_argument_group('pgan', 'parameters used for pgan walk')
        group.add_argument('--dset', default="celebahq", help="which dataset to use for pretrained pgan")

        self.initialized = True
        return self.parser

    def print_options(self, opt):
        opt_dict = OrderedDict()
        message = ''
        message += '----------------- Options ---------------\n'
        # top level options
        grouped_k = []
        for k, v in sorted(vars(opt).items()):
            if isinstance(v, argparse.Namespace):
                grouped_k.append((k, v))
                continue
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            opt_dict[k] = v
        # grouped options
        for k, v in grouped_k:
            message += '{} '.format(k).ljust(20, '-')
            message += '\n'
            opt_dict[k] = OrderedDict()
            for k1, v1 in sorted(vars(v).items()):
                comment = ''
                default = self.parser.get_default(k1)
                if v1 != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<30}{}\n'.format(str(k1), str(v1), comment)
                opt_dict[k][k1] = v1
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if hasattr(opt, 'output_dir'):
                expr_dir = opt.output_dir
        else:
            expr_dir ='./'
        os.makedirs(expr_dir, exist_ok=True)

        if not opt.overwrite_config:
            assert(not os.path.isfile(os.path.join(expr_dir, 'opt.txt'))), 'config file exists, use --overwrite_config'
            assert(not os.path.isfile(os.path.join(expr_dir, 'opt.yml'))), 'config file exists, use --overwrite_config'

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        file_name = os.path.join(expr_dir, 'opt.yml')
        with open(file_name, 'wt') as opt_file:
            opt_dict['overwrite_config'] = False
            yaml.dump(opt_dict, opt_file, default_flow_style=False)

    def _flatten_to_toplevel(self, data):
        args = {}
        for k, v in data.items():
            if isinstance(v, dict):
                args.update(self._flatten_to_toplevel(v))
            else:
                args[k] = v
        return args

    def parse(self, print_opt=True):
        ''' use update_fn() to do additional modifications on args
            before printing
        '''
        # initialize parser with basic options
        if not self.initialized:
            self.initialize()

        # parse options
        opt = self.parser.parse_args()

        # get arguments specified in config file
        if opt.config_file:
            data = yaml.load(opt.config_file, Loader=yaml.FullLoader)
            data = self._flatten_to_toplevel(data)
        else:
            data = {}

        # determine which options were specified
        # explicitly with command line args
        option_strings = {}
        for action_group in self.parser._action_groups:
            for action in action_group._group_actions:
                for option in action.option_strings:
                    option_strings[option] = action.dest
        specified_options = set([option_strings[x] for x in
                                 sys.argv if x in option_strings])

        # make hierarchical namespace wrt groups
        # positional and optional arguments in toplevel
        args = {}
        for group in self.parser._action_groups:
        # by default, take the result from argparse
            # unless was specified in config file and not in command line
            group_dict={a.dest: data[a.dest] if a.dest in data
                        and a.dest not in specified_options
                        else getattr(opt, a.dest, None)
                        for a in group._group_actions}
            if group.title == 'positional arguments' or \
               group.title == 'optional arguments':
                args.update(group_dict)
            else:
                args[group.title] = argparse.Namespace(**group_dict)

        opt = argparse.Namespace(**args)
        delattr(opt, 'config_file')

        # output directory
        if opt.name:
            output_dir = opt.name
        else:
            output_dir = '_'.join([opt.model, opt.transform, opt.walk_type,
                                    'lr'+str(opt.learning_rate), opt.loss])
            if opt.model == 'biggan':
                subopt = opt.biggan
                if subopt.category:
                    output_dir += '_cat{}'.format(subopt.category)
            elif opt.model == 'stylegan':
                subopt = opt.stylegan
                output_dir += '_{}'.format(subopt.dataset)
                output_dir += '_{}'.format(subopt.latent)
            elif opt.model == 'pgan':
                subopt = opt.pgan
                output_dir += '_{}'.format(subopt.dset)
            if opt.walk_type.startswith('NN'):
                subopt = opt.nn
                if subopt.eps:
                    output_dir += '_eps{}'.format(subopt.eps)
                if subopt.num_steps:
                    output_dir += '_nsteps{}'.format(subopt.num_steps)
            if opt.transform.startswith('color') and opt.color.channel is not None:
                output_dir += '_chn{}'.format(opt.color.channel)


        if opt.suffix:
            output_dir += opt.suffix
        if opt.prefix:
            output_dir = opt.prefix + output_dir

        opt.output_dir = os.path.join(opt.models_dir, output_dir)


        # write the configurations to disk
        if print_opt:
            self.print_options(opt)

        self.opt = opt
        return opt
