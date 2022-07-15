import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import pandas as pd
import torch
import util.util as util
import numpy as np
import torchvision.transforms.functional as F

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase + '_resize')
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')
        self.dir_SP = opt.dirSem
        self.SP_input_nc = opt.SP_input_nc

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)
        self.use_BPD = self.opt.use_BPD

        self.finesize = opt.fineSize

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size - 1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name)
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy')

        P2_path = os.path.join(self.dir_P, P2_name)
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy')

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path)
        BP2_img = np.load(BP2_path)

        if self.use_BPD:
            BPD1_img = util.draw_dis_from_map(BP1_img)[0]
            BPD2_img = util.draw_dis_from_map(BP2_img)[0]

        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0, 1)

            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :])
                BP2_img = np.array(BP2_img[:, ::-1, :])

            BP1 = torch.from_numpy(BP1_img).float()
            BP1 = BP1.transpose(2, 0)
            BP1 = BP1.transpose(2, 1)

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0)
            BP2 = BP2.transpose(2, 1)

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)
        else:
            BP1 = torch.from_numpy(BP1_img).float()
            BP1 = BP1.transpose(2, 0)
            BP1 = BP1.transpose(2, 1)

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0)
            BP2 = BP2.transpose(2, 1)

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)
            if self.use_BPD:
                BPD1 = torch.from_numpy(BPD1_img).float()
                BPD1 = BPD1.transpose(2, 0)
                BPD1 = BPD1.transpose(2, 1)

                BPD2 = torch.from_numpy(BPD2_img).float()
                BPD2 = BPD2.transpose(2, 0)
                BPD2 = BPD2.transpose(2, 1)


        SP1_name = self.split_name_sementic3(P1_name, 'semantic_merge3')
        SP2_name = self.split_name_sementic3(P2_name, 'semantic_merge3')
        SP1_path = os.path.join(self.dir_SP, SP1_name)
        SP1_path = SP1_path[:-4] + '.png'
        SP1_data = Image.open(SP1_path)
        SP1_data = np.array(SP1_data)
        SP2_path = os.path.join(self.dir_SP, SP2_name)
        SP2_path = SP2_path[:-4] + '.png'
        SP2_data = Image.open(SP2_path)
        SP2_data = np.array(SP2_data)
        SP1 = np.zeros((self.SP_input_nc, self.finesize[0], self.finesize[1]), dtype='float32')
        SP2 = np.zeros((self.SP_input_nc, self.finesize[0], self.finesize[1]), dtype='float32')
        SP1_20 = np.zeros((20, self.finesize[0], self.finesize[1]), dtype='float32')
        SP2_20 = np.zeros((20, self.finesize[0], self.finesize[1]), dtype='float32')
        nc = 20
        for id in range(nc):
            SP1_20[id] = (SP1_data == id).astype('float32')
            SP2_20[id] = (SP2_data == id).astype('float32')
        SP1[0] = SP1_20[0]
        SP1[1] = SP1_20[9] + SP1_20[12]
        SP1[2] = SP1_20[2] + SP1_20[1]
        SP1[3] = SP1_20[3]
        SP1[4] = SP1_20[13] + SP1_20[4]
        SP1[5] = SP1_20[5] + SP1_20[6] + SP1_20[7] + SP1_20[10] + SP1_20[11]
        SP1[6] = SP1_20[14] + SP1_20[15]
        SP1[7] = SP1_20[8] + SP1_20[16] + SP1_20[17] + SP1_20[18] + SP1_20[19]

        SP2[0] = SP2_20[0]
        SP2[1] = SP2_20[9] + SP2_20[12]
        SP2[2] = SP2_20[2] + SP2_20[1]
        SP2[3] = SP2_20[3]
        SP2[4] = SP2_20[13] + SP2_20[4]
        SP2[5] = SP2_20[5] + SP2_20[6] + SP2_20[7] + SP2_20[10] + SP2_20[11]
        SP2[6] = SP2_20[14] + SP2_20[15]
        SP2[7] = SP2_20[8] + SP2_20[16] + SP2_20[17] + SP2_20[18] + SP2_20[19]


        if self.use_BPD:
            return {'P1': P1, 'BP1': BP1, 'SP1': SP1, 'BPD1': BPD1,
                    'P2': P2, 'BP2': BP2, 'SP2': SP2, 'BPD2': BPD2,
                    'P1_path': P1_name, 'P2_path': P2_name}
        else:
            return {'P1': P1, 'BP1': BP1, 'SP1': SP1,
                    'P2': P2, 'BP2': BP2, 'SP2': SP2,
                    'P1_path': P1_name, 'P2_path': P2_name}

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'


    def split_name_sementic3(self, str, type):
        list = []
        list.append(type)
        list.append(str)

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head

