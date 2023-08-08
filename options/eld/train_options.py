from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--adaptive_loss', action="store_true", help='whether to use a learned weight of loss for different stages')
        # self.parser.add_argument('--used_ratio', type=list, nargs="+", default=None, help="only works for train_real.py")
        self.parser.add_argument('--auxloss', action="store_true")
        # self.parser.add_argument('--patchSize', default=512, help='input patch size') # 因为数据集是lmdb 所以没有实现不同patch size训练   
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')  
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')          
        self.parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        self.parser.add_argument('--loss', type=str, default='l1', help='pixel loss type')
        self.parser.add_argument('--noise', type=str, default='g', help='noise model to use')
        self.parser.add_argument('--exclude', type=int, default=None, help='camera exclude id')
        self.parser.add_argument('--continuous_noise', action="store_true", help='ratio sampled from 1 to 300 instead of 100 to 300')
        
        # Training options for real noise
        self.parser.add_argument("--syn_noise", action="store_true")
        
        self.isTrain = True
