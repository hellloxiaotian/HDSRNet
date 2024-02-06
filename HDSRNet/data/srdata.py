import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            # print('srdata-path_bin0:', path_bin)  # /data/zxy/datasets/DIV2K/bin
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            # print('srdata-self.dir_hr1', self.dir_hr)  # /data/zxy/datasets/DIV2K/DIV2K_train_HR
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            # print('srdata-self.dir_hr2', self.dir_hr)  # /data/zxy/datasets/DIV2K/DIV2K_train_HR
            for s in self.scale:
                # print('srdata-self.dir_lr1', self.dir_lr)  # /data/zxy/datasets/DIV2K/DIV2K_train_LR_bicubic
                # print('srdata-path_bin', path_bin)  # /data/zxy/datasets/DIV2K/bin
                # print('srdata-self.apath', self.apath)  # /data/zxy/datasets/DIV2K
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
                # print('srdata-self.dir_lr2', self.dir_lr)  # /data/zxy/datasets/DIV2K/DIV2K_train_LR_bicubic

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for i, ll in enumerate(list_lr):
                # print('i', i)
                # print('ll', ll)
                b = ll.replace(self.apath, path_bin)
                # print('b1', b)
                b = b.replace(self.ext[1], '.pt')
                # print('self.ext[1]', self.ext[1])
                # print('self.images_lr', self.images_lr)
                # self.images_lr[i].append(b)
                self.images_lr.append(b)
                self._check_and_load(args.ext, ll, b, verbose=True)
                '''
                for l in ll:
                    print('l', l)
                    print('path_bin', path_bin)
                    print('self.apath', self.apath)
                    b = l.replace(self.apath, path_bin)
                    print('b1', b)
                    b = b.replace(self.ext[1], '.pt')
                    print('self.ext[1]', self.ext[1])
                    print('b2', b)
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)
                '''

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        print('srdata-scan()')
        if self.name in ['DIV2K']:
            names_lr = sorted(
                # glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
                glob.glob(os.path.join(self.dir_lr, 'X{}'.format(self.scale), '*' + self.ext[0]))
            )
        else:
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
                # glob.glob(os.path.join(self.dir_lr, 'X{}'.format(self.scale), '*' + self.ext[0]))
            )

        # names_lr = sorted(
        #     # glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        #     glob.glob(os.path.join(self.dir_lr, 'X{}'.format(self.scale), '*' + self.ext[0]))
        # )
        '''
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    # self.dir_lr, 'X{}/{}x{}{}'.format(
                    #     s, filename, s, self.ext[1]
                    self.dir_lr, '{}{}'.format(
                        filename, self.ext[1]
                    )
                ))
        '''

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        # print('dir_data111:', dir_data)  # This code will not run.
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        # print('idx0', idx)  # hr图像list下标从0开始
        f_hr = self.images_hr[idx]
        # print('idx1', idx+1)  # lr图像list下标从1开始
        # print('srdata-self.images_lr:', self.images_lr)
        # print('self.images_hr:', self.images_hr)  # '/data/zxy/datasets/DIV2K/bin/DIV2K_train_HR/0656.pt',
        # print('idx_scale:', self.idx_scale)  # 4
        # print('idx:', idx)  # index number of image. 1,2,3,4
        # f_lr = self.images_lr[idx]
        # print('self.images_lr', self.images_lr)
        # print('self.name', self.name)
        if self.name in ['DIV2K']:
            f_lr = self.images_lr[idx+1]
        else:
            # print('self.images_lr', self.images_lr)
            f_lr = self.images_lr[idx]


        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                # print('f_hr:', f_hr)
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                # print('srdata-f_lr', f_lr)  # 执行这里
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        # print('self.idx_scale', self.idx_scale)
        # print('srdata-get_patch-scale')
        # print('srdata-self.scale', self.scale)
        # print('srdata-self.idx_scale', self.idx_scale)
        # scale = self.scale[self.idx_scale]
        scale = self.scale[0]
        # print('srdata-scale', scale)  # 4
        if self.train:
            # print('srdata-lr:', lr.shape)
            # print('srdata-hr:', hr.shape)
            # print('self.args.patch_size', self.args.patch_size)  # 64
            # print('self.input_large', self.input_large)  # False
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            # print('self.args.no_augment', self.args.no_augment)  # False
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

