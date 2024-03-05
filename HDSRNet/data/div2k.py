import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
            print('data_range', data_range)
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        # print('names_lr', names_lr)
        ### The Path of our data is as bellow:
        # '/data/zxy/datasets/DIV2K/DIV2K_train_LR_bicubic/X4/0900x4.png'

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        # print('dir_hr:', self.dir_hr)
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        # self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic_all')
        # self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic/X4')
        # self.dir_lr = os.path.join(self.apath, 'DIV2K_train_HR')

        ### test code to print your data path
        # print('dir_lr:', self.dir_lr)
        if self.input_large: self.dir_lr += 'L'

