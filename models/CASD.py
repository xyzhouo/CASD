import torch.nn as nn
import functools
import torch
import torch.nn.functional as F

import os
import torchvision.models.vgg as models
from torch.nn.parameter import Parameter

from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import functools


# Moddfied with AdINGen
class ADGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(ADGen, self).__init__()

        # style encoder
        input_dim = 3
        self.SP_input_nc = 8
        self.enc_style = VggStyleEncoder(3, input_dim, dim, int(style_dim / self.SP_input_nc), norm='none', activ=activ,
                                         pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(layers=2, ngf=64, img_f=512)

        input_dim = 3
        self.dec = Decoder(style_dim, mlp_dim, n_downsample, n_res, 256, input_dim,
                           self.SP_input_nc, res_norm='adain', activ=activ, pad_type=pad_type)

    def forward(self, img_A, img_B, sem_B):
        content = self.enc_content(img_A)
        style = self.enc_style(img_B, sem_B)
        images_recon = self.dec(content, style)
        return images_recon


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class VggStyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(VggStyleEncoder, self).__init__()
        # self.vgg = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('/home/haihuam/CASD-main/dataset/fashion/vgg19-dcbb9e9d.pth'))
        self.vgg = vgg19.features

        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)  # 3->64
        dim = dim * 2
        self.conv2 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 128->128
        dim = dim * 2
        self.conv3 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 256->256
        dim = dim * 2
        self.conv4 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 512->512
        dim = dim * 2

        self.model0 = []
        self.model0 += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model0 = nn.Sequential(*self.model0)

        self.AP = []
        self.AP += [nn.AdaptiveAvgPool2d(1)]
        self.AP = nn.Sequential(*self.AP)
        self.output_dim = dim

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1'}
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def texture_enc(self, x):
        sty_fea = self.get_features(x, self.vgg)
        x = self.conv1(x)
        x = torch.cat([x, sty_fea['conv1_1']], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, sty_fea['conv2_1']], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, sty_fea['conv3_1']], dim=1)
        x = self.conv4(x)
        x = torch.cat([x, sty_fea['conv4_1']], dim=1)
        x0 = self.model0(x)
        return x0

    def forward(self, x, sem):

        codes = self.texture_enc(x)
        segmap = F.interpolate(sem, size=codes.size()[2:], mode='nearest')

        bs = codes.shape[0]
        hs = codes.shape[2]
        ws = codes.shape[3]
        cs = codes.shape[1]
        f_size = cs

        s_size = segmap.shape[1]
        codes_vector = torch.zeros((bs, s_size, cs), dtype=codes.dtype, device=codes.device)

        for i in range(bs):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,
                                              component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature
                else:
                    tmpmean, tmpstd = calc_mean_std(
                        codes[i].reshape(1, codes[i].shape[0], codes[i].shape[1], codes[i].shape[2]))
                    codes_vector[i][j] = tmpmean.squeeze()


        return codes_vector.view(bs, -1).unsqueeze(2).unsqueeze(3)


class ContentEncoder(nn.Module):
    def __init__(self, layers=2, ngf=64, img_f=512, use_spect = False, use_coord = False):
        super(ContentEncoder, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type='instance')
        nonlinearity = get_nonlinearity_layer(activation_type='LeakyReLU')
        self.ngf = ngf
        self.img_f = img_f
        self.block0 = EncoderBlock(30, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(self.layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), self.img_f//self.ngf)
            block = EncoderBlock(self.ngf*mult_prev, self.ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.model0 = []
        self.model0 += [norm_layer(128)]
        self.model0 += [nonlinearity]
        self.model0 += [nn.Conv2d(128, 256, 1, 1, 0)]
        self.model0 = nn.Sequential(*self.model0)

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.model0(out)
        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = torch.reshape(x, (b, c, h, w))
        return x



class Decoder(nn.Module):
    def __init__(self, style_dim, mlp_dim, n_upsample, n_res, dim, output_dim, SP_input_nc, res_norm='adain',
                 activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.softmax_style = nn.Softmax(dim=2)
        self.SP_input_nc = SP_input_nc
        self.model0 = []
        self.model1 = []
        self.model2 = []
        self.n_res = n_res

        self.mlp = MLP(style_dim, n_res * dim * 4, mlp_dim, 3, norm='none', activ=activ)
        self.fc = LinearBlock(style_dim, style_dim, norm='none', activation=activ)

        # AdaIN residual blocks
        self.model0_0 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_0 = nn.Sequential(*self.model0_0)
        self.model0_1 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_1 = nn.Sequential(*self.model0_1)
        self.model0_2 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_2 = nn.Sequential(*self.model0_2)
        self.model0_3 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_3 = nn.Sequential(*self.model0_3)
        self.model0_4 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_4 = nn.Sequential(*self.model0_4)
        self.model0_5 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_5 = nn.Sequential(*self.model0_5)
        self.model0_6 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_6 = nn.Sequential(*self.model0_6)
        self.model0_7 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_7 = nn.Sequential(*self.model0_7)
        # upsampling blocks
        for i in range(n_upsample):
            self.model1 += [nn.Upsample(scale_factor=2),
                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.model1 = nn.Sequential(*self.model1)
        # use reflection padding in the last conv layer
        self.model2 += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model2 = nn.Sequential(*self.model2)
        # attention parameter

        self.gamma3_1 = nn.Parameter(torch.zeros(1))
        self.gamma3_2 = nn.Parameter(torch.zeros(1))
        self.gamma3_3 = nn.Parameter(torch.zeros(1))
        self.gamma3_style_sa = nn.Parameter(torch.zeros(1))
        in_dim = int(style_dim / self.SP_input_nc)
        self.value3_conv_sa = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.LN_3_style = ILNKVT(256)
        self.LN_3_pose = ILNQT(256)
        self.LN_3_pose_0 = ILNQT(256)
        self.query3_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key3_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value3_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query3_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=self.SP_input_nc, kernel_size=1)

        self.gamma4_1 = nn.Parameter(torch.zeros(1))
        self.gamma4_2 = nn.Parameter(torch.zeros(1))
        self.gamma4_3 = nn.Parameter(torch.zeros(1))
        self.gamma4_style_sa = nn.Parameter(torch.zeros(1))
        in_dim = int(style_dim / self.SP_input_nc)
        self.value4_conv_sa = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.LN_4_style = ILNKVT(256)
        self.LN_4_pose = ILNQT(256)
        self.LN_4_pose_0 = ILNQT(256)
        self.query4_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key4_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value4_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query4_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=self.SP_input_nc, kernel_size=1)

        self.FFN3_1 = FFN(256)
        self.FFN4_1 = FFN(256)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, style):
        # fusion module
        style_fusion = self.fc(style.view(style.size(0), -1))
        adain_params = self.mlp(style_fusion)
        adain_params = torch.split(adain_params, int(adain_params.shape[1] / self.n_res), 1)

        x_0 = x
        x = self.model0_0([x, adain_params[0]])
        x = self.model0_1([x, adain_params[1]])
        x = self.model0_2([x, adain_params[2]])
        x = self.model0_3([x, adain_params[3]])

        x3, enerrgy_sum3 = self.styleatt(x, x_0, style, self.gamma3_1, self.gamma3_2, self.gamma3_3, \
                                                 self.gamma3_style_sa, self.value3_conv_sa, \
                                                 self.LN_3_style, self.LN_3_pose, self.LN_3_pose_0, \
                                                 self.query3_conv, self.key3_conv, self.value3_conv, self.query3_conv_0, \
                                                 self.FFN3_1)

        x_, enerrgy_sum4 = self.styleatt(x3, x_0, style, self.gamma4_1, self.gamma4_2, self.gamma4_3, \
                                                 self.gamma4_style_sa, self.value4_conv_sa, \
                                                 self.LN_4_style, self.LN_4_pose, self.LN_4_pose_0, \
                                                 self.query4_conv, self.key4_conv, self.value4_conv, self.query4_conv_0, \
                                                 self.FFN4_1)

        x = self.model0_4([x_0, x_])
        x = self.model0_5([x, x_])
        x = self.model0_6([x, x_])
        x = self.model0_7([x, x_])
        x = self.model1(x)
        return self.model2(x), [enerrgy_sum3, enerrgy_sum4]

    def styleatt(self, x, x_0, style, gamma1, gamma2, gamma3, gamma_style_sa, value_conv_sa, ln_style, ln_pose,
                 ln_pose_0, query_conv, key_conv, value_conv, query_conv_0, ffn1):
        B, C, H, W = x.size()
        B, Cs, _, _ = style.size()
        K = self.SP_input_nc
        style = style.view((B, K, int(Cs / K))) # [B,K,C]

        x = ln_pose(x) # [B,C,H,W]
        style = ln_style(style.permute(0, 2, 1)) # [B,C,K]
        x_0 = ln_pose_0(x_0)

        style = style.permute(0, 2, 1) # [B,K,C]
        style_sa_value = torch.squeeze(value_conv_sa(torch.unsqueeze(style.permute(0, 2, 1), 3)), 3) # [B,C,K]
        self_att = self.softmax(torch.bmm(style, style.permute(0, 2, 1))) + 1e-8  # [B,K,K]
        self_att = self_att / torch.sum(self_att, dim=2, keepdim=True)
        style_ = torch.bmm(self_att, style_sa_value.permute(0, 2, 1))
        style = style + gamma_style_sa * style_  # [B,K,C]

        style = style.permute(0, 2, 1)  #[B,C,K]
        x_query = query_conv(x)
        style_key = torch.squeeze(key_conv(torch.unsqueeze(style, 3)).permute(0, 2, 1, 3), 3)
        style_value = torch.squeeze(value_conv(torch.unsqueeze(style, 3)), 3)

        energy_0 = query_conv_0(x_0).view((B, K, H * W))
        energy = torch.bmm(style_key.detach(), x_query.view(B, C, -1))
        enerrgy_sum = energy_0 + energy
        attention = self.softmax_style(enerrgy_sum) + 1e-8
        attention = attention / torch.sum(attention, dim=1, keepdim=True)

        out = torch.bmm(style_value, attention)
        out = out.view(B, C, H, W)
        out = gamma1 * out + x
        out = out + gamma3 * ffn1(out)

        return out, torch.reshape(enerrgy_sum, (B, K, H, W))


class ILNKVT(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2], keepdim=True), torch.var(input, dim=[2], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1], keepdim=True), torch.var(input, dim=[1], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1) + self.beta.expand(input.shape[0], -1, -1)

        return out

class ILNQT(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1], keepdim=True), torch.var(input, dim=[1], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock_myDFNM(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock_myDFNM, self).__init__()

        model1 = []
        model2 = []
        model1 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model2 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        models1 = []
        models1 += [Conv2dBlock(dim, dim, 3, 1, 1, norm='in', activation='relu', pad_type=pad_type)]
        models1 += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)]
        models2 = []
        models2 += [Conv2dBlock(dim, dim, 3, 1, 1, norm='in', activation='relu', pad_type=pad_type)]
        models2 += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.models1 = nn.Sequential(*models1)
        self.models2 = nn.Sequential(*models2)

    def forward(self, x):
        style = x[1]
        style1 = self.models1(style)
        style2 = self.models2(style)
        residual = x[0]
        out = self.model1([x[0], style1])
        out = self.model2([out, style2])
        out += residual

        return out


class ResBlock_my(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock_my, self).__init__()

        model1 = []
        model2 = []
        model1 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model2 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, x):
        style = x[1]
        style1, style2 = torch.split(style, int(style.shape[1] / 2), 1)
        residual = x[0]
        out = self.model1([x[0], style1])
        out = self.model2([out, style2])
        out += residual
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock_my(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock_my, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'spade':
            self.norm = SPADE()
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        style = x[1]
        x = x[0]
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm([x, style])
        if self.activation:
            x = self.activation(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################
class SPADE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        style = x[1]
        x = x[0]
        # Part 1. generate parameter-free normalized activations
        x_mean = torch.mean(x, (0, 2, 3), keepdim=True)
        x_var = torch.var(x, (0, 2, 3), keepdim=True)
        normalized = (x - x_mean) / (x_var + 1e-6)

        # Part 2. produce scaling and bias conditioned on semantic map
        gamma, beta = torch.split(style, int(style.size(1) / 2), 1)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        style = x[1]
        self.weight, self.bias = torch.split(style, int(style.shape[1] / 2), 1)
        x = x[0]
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'adain':
        norm_layer = functools.partial(ADAIN)
    elif norm_type == 'spade':
        norm_layer = functools.partial(SPADE, config_text='spadeinstance3x3')
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer

def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module



class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()


        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = coord_conv(input_nc,  output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc),  nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2,)

    def forward(self, x):
        out = self.model(x)
        return out
