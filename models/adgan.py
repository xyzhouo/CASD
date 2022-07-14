from torch.autograd import Variable
import numpy as np
import torch
import os
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from losses.CX_style_loss import CXLoss
from .vgg_SC import VGG, VGGLoss
from losses.lpips.lpips import LPIPS



class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.use_AMCE = opt.use_AMCE
        self.use_BPD = opt.use_BPD
        self.SP_input_nc = opt.SP_input_nc
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size[0], size[1])
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size[0], size[1])
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size[0], size[1])
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size[0], size[1])
        self.input_SP1_set = self.Tensor(nb, opt.SP_input_nc, size[0], size[1])
        self.input_SP2_set = self.Tensor(nb, opt.SP_input_nc, size[0], size[1])
        if self.use_BPD:
            self.input_BPD1_set = self.Tensor(nb, opt.BPD_input_nc, size[0], size[1])
            self.input_BPD2_set = self.Tensor(nb, opt.BPD_input_nc, size[0], size[1])


        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc + (opt.BPD_input_nc+opt.BPD_input_nc if self.use_BPD else 0)]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc + (opt.BPD_input_nc if self.use_BPD else 0), opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if len(opt.gpu_ids) > 1:
                self.load_VGG(self.netG.module.enc_style.vgg)
            else:
                self.load_VGG(self.netG.enc_style.vgg)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)


            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Excption('Unsurportted type of L1!')

            if opt.use_cxloss:
                self.CX_loss = CXLoss(sigma=0.5)
                if torch.cuda.is_available():
                    self.CX_loss.cuda()
                self.vgg = VGG()
                self.vgg.load_state_dict(torch.load(os.path.abspath(opt.dataroot) + '/vgg_conv.pth'))
                for param in self.vgg.parameters():
                    param.requires_grad = False
                if torch.cuda.is_available():
                    self.vgg.cuda()

            if opt.use_lpips:
                self.lpips_loss = LPIPS(net_type='vgg').cuda().eval()

            if self.use_AMCE:
                self.AM_CE_loss = torch.nn.CrossEntropyLoss()
                if torch.cuda.is_available():
                    self.AM_CE_loss.cuda()


            self.Vggloss = VGGLoss().cuda().eval()


            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

        
    def set_input(self, input):
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)
        
        if self.use_BPD:
            input_BPD1, input_BPD2 = input['BPD1'], input['BPD2']
            self.input_BPD1_set.resize_(input_BPD1.size()).copy_(input_BPD1)
            self.input_BPD2_set.resize_(input_BPD2.size()).copy_(input_BPD2)
            
        input_SP1 = input['SP1']
        self.input_SP1_set.resize_(input_SP1.size()).copy_(input_SP1)
        if self.use_AMCE:
            input_SP2 = input['SP2']
            self.input_SP2_set.resize_(input_SP2.size()).copy_(input_SP2)

        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        self.person_paths = input['P1_path'][0]


    def forward(self):
        
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)
        
        if self.use_BPD:
            self.input_BPD1 = Variable(self.input_BPD1_set)
            self.input_BPD2 = Variable(self.input_BPD2_set)
            
        self.input_SP1 = Variable(self.input_SP1_set)
        self.input_SP2 = Variable(self.input_SP2_set)

        if self.use_BPD:
            self.fake_p2, self.fake_sp2 = self.netG(torch.cat([self.input_BP2, self.input_BPD2], 1), self.input_P1, self.input_SP1)
        else:
            self.fake_p2, self.fake_sp2 = self.netG(self.input_BP2, self.input_P1, self.input_SP1)


    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)
        
        if self.use_BPD:
            self.input_BPD1 = Variable(self.input_BPD1_set)
            self.input_BPD2 = Variable(self.input_BPD2_set)
            
        self.input_SP1 = Variable(self.input_SP1_set)
        self.input_SP2 = Variable(self.input_SP2_set)


        if self.use_BPD:
            self.fake_p2, self.fake_sp2 = self.netG(torch.cat([self.input_BP2, self.input_BPD2], 1), self.input_P1, self.input_SP1)
        else:
            self.fake_p2, self.fake_sp2 = self.netG(self.input_BP2, self.input_P1, self.input_SP1)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_person_paths(self):
        return self.person_paths


    def backward_G(self):
        if self.opt.with_D_PB:
            if self.use_BPD:
                pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2, self.input_BPD2), 1))
            else:
                pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # CX loss
        if self.opt.use_cxloss:
            style_layer = ['r32', 'r42']
            vgg_style = self.vgg(self.input_P2, style_layer)
            vgg_fake = self.vgg(self.fake_p2, style_layer)
            cx_style_loss = 0

            for i, val in enumerate(vgg_fake):
                cx_style_loss += self.CX_loss(vgg_style[i], vgg_fake[i])
            cx_style_loss *= self.opt.lambda_cx

        pair_cxloss = cx_style_loss

        if self.opt.use_lpips:
            lpips_loss = self.lpips_loss(self.fake_p2, self.input_P2)
            lpips_loss *= self.opt.lambda_lpips
            pair_lpips_loss = lpips_loss

        # Attention Map Cross Entropy loss
        if self.use_AMCE:
            up_ = torch.nn.Upsample(scale_factor=4, mode='bilinear')
            if isinstance(self.fake_sp2,list):
                AMCE_loss = 0
                B, C, H, W = self.input_SP2.shape
                for i in range(len(self.fake_sp2)):
                    logits = up_(self.fake_sp2[i])
                    logits = torch.reshape(logits.permute(0,2,3,1), (B*H*W, C))
                    labels = torch.argmax(torch.reshape(self.input_SP2.permute(0,2,3,1), (B*H*W, C)), 1)
                    AMCE_loss += self.AM_CE_loss(logits, labels)

                AMCE_loss *= self.opt.lambda_AMCE
                pair_AMCE_loss = AMCE_loss
            else:
                logits = up_(self.fake_sp2)
                B, C, H, W = self.input_SP2.shape
                logits = torch.reshape(logits.permute(0,2,3,1), (B*H*W, C))
                labels = torch.argmax(torch.reshape(self.input_SP2.permute(0,2,3,1), (B*H*W, C)), 1)
                AMCE_loss = self.AM_CE_loss(logits, labels)
                AMCE_loss *= self.opt.lambda_AMCE
                pair_AMCE_loss = AMCE_loss

        self.opt.lambda_style = 200
        self.opt.lambda_content = 0.5
        loss_content_gen, loss_style_gen = self.Vggloss(self.fake_p2, self.input_P2)
        pair_style_loss = loss_style_gen*self.opt.lambda_style
        pair_content_loss = loss_content_gen*self.opt.lambda_content



        # L1 loss
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].data
            self.loss_perceptual = losses[2].data

        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A

        pair_L1loss = self.loss_G_L1

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN


        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        if self.opt.use_cxloss:
            pair_loss = pair_loss + pair_cxloss
        if self.opt.use_AMCE:
            pair_loss = pair_loss + pair_AMCE_loss
        if self.opt.use_lpips:
            pair_loss = pair_loss + pair_lpips_loss

        pair_loss = pair_loss + pair_content_loss
        pair_loss = pair_loss + pair_style_loss

        pair_loss.backward()

        self.pair_L1loss = pair_L1loss.data
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.data

        if self.opt.use_cxloss:
            self.pair_cxloss = pair_cxloss.data

        if self.opt.use_lpips:
            self.pair_lpips_loss = pair_lpips_loss.data
        if self.opt.use_AMCE:
            self.pair_AMCE_loss = pair_AMCE_loss.data

        self.pair_content_loss = pair_content_loss.data
        self.pair_style_loss = pair_style_loss.data


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        if self.use_BPD:
            real_PB = torch.cat((self.input_P2, self.input_BP2, self.input_BPD2), 1)
            fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2, self.input_BPD2), 1).data )
        else:
            real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
            fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)

        self.loss_D_PB = loss_D_PB.data

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)

        self.loss_D_PP = loss_D_PP.data


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP or self.opt.with_D_PS:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual

        if self.opt.use_cxloss:
            ret_errors['CXLoss'] = self.pair_cxloss
        if self.opt.use_lpips:
            ret_errors['lpips'] = self.pair_lpips_loss
        if self.opt.use_AMCE:
            ret_errors['AMCE'] = self.pair_AMCE_loss

        ret_errors['content'] = self.pair_content_loss
        ret_errors['style'] = self.pair_style_loss

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]


        if self.use_BPD:
            input_BPD1 = util.draw_dis_from_map(self.input_BP1.data)[1]
            input_BPD1 = (np.repeat(np.expand_dims(input_BPD1, -1), 3, -1)*255).astype('uint8')
            input_BPD2 = util.draw_dis_from_map(self.input_BP2.data)[1]
            input_BPD2 = (np.repeat(np.expand_dims(input_BPD2, -1), 3, -1)*255).astype('uint8')


        fake_p2 = util.tensor2im(self.fake_p2.data)
        
        if self.use_BPD:
            vis = np.zeros((height, width*7, 3)).astype(np.uint8) #h, w, c
            vis[:, :width, :] = input_P1
            vis[:, width:width*2, :] = input_BP1
            vis[:, width*2:width*3, :] = input_BPD1
            vis[:, width*3:width*4, :] = input_P2
            vis[:, width*4:width*5, :] = input_BP2
            vis[:, width*5:width*6, :] = input_BPD2
            vis[:, width*6:width*7, :] = fake_p2
        else:
            vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
            vis[:, :width, :] = input_P1
            vis[:, width:width*2, :] = input_BP1
            vis[:, width*2:width*3, :] = input_P2
            vis[:, width*3:width*4, :] = input_BP2
            vis[:, width*4:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals


    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)



     
