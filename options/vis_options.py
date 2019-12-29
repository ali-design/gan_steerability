import argparse
import oyaml as yaml

class VisOptions():
    def __init__(self):
        self.initialized = False
        self.parser = parser = argparse.ArgumentParser("Visualization Parser")

    def initialize(self):
        parser = self.parser
        parser.add_argument('weight_path', help="path to saved weights")
        parser.add_argument('config_file', type=argparse.FileType(mode='r'), help="configuration yml file")
        parser.add_argument("--gpu", default="", type=str, help='GPUs to use (leave blank for CPU only)')
        parser.add_argument('--noise_seed', type=int, default=0, help="noise seed for z samples")
        parser.add_argument('--output_dir', help="where to save output; if specified, overrides output_dir in config file")

        # biggan options
        group = parser.add_argument_group('biggan', 'parameters used for biggan model')
        group.add_argument('--category', type=int, default=0, help='categories to visualize')
        group.add_argument('--truncation', type=float, default=1.0,
                            help='truncation for z samples')


        self.initialized = True
        return self.parser


    def parse(self):
        # initialize parser with basic options
        if not self.initialized:
            self.initialize()

        # parse options
        opt = self.parser.parse_args()

        # get arguments specified in config file
        # and convert to a namespace
        data = yaml.load(opt.config_file, Loader=yaml.FullLoader)
        for k,v in data.items():
            if isinstance(v, dict):
                data[k] = argparse.Namespace(**v)
        data = argparse.Namespace(**data)

        self.opt = opt
        self.data = data
        return opt, data
