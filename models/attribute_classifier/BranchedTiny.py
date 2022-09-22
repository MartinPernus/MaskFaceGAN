
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def define_net(w, img_size):
    # Convolutional part
    conv_layout = [w, w, 'M', w, w, 'M',  w, w,  
                    w, 'M', w, w, w, 'M', w,
                    w, w, 'M']

    conv_cfg = {'01-06': {'n_layers': 1 , 'kwargs': dict(layout=conv_layout[0:8], in_channels=3, out_channels=w)},
                '06-11':  {'n_layers': 2 , 'kwargs': dict(layout=conv_layout[8:15], in_channels=w, out_channels=w)},
                '11-12': {'n_layers': 3, 'kwargs': dict(layout=conv_layout[15:16], in_channels=w, out_channels=w)},
                '12-13': {'n_layers': 10, 'kwargs': dict(layout=conv_layout[16:18], in_channels=w, out_channels=w)}}

    conv_layers = {k: [] for k in conv_cfg}
    for node, cfg in conv_cfg.items():
        for _ in range(cfg['n_layers']):
            conv_layers[node] += [make_conv_layers(**cfg['kwargs'])]
        conv_layers[node] = nn.ModuleList(conv_layers[node])
    conv_layers = nn.ModuleDict(conv_layers)

    # Fully connected part
    M = get_flattened_dim(conv_layers.values(), img_size)
    fc_cfg = {'13-14': {'n_layers': 18, 'kwargs': dict(in_channels=M, out_channels=2*w, relu=True)},
              '14-15': {'n_layers': 33, 'kwargs': dict(in_channels=2*w, out_channels=2*w, relu=True)},
              '15-16': {'n_layers': 40, 'kwargs': dict(in_channels=2*w, out_channels=1, relu=False)}}

    fc_layers = {k: [] for k in fc_cfg}
    for node, cfg in fc_cfg.items():
        for _ in range(cfg['n_layers']):
            fc_layers[node] += [make_fc_layers(**cfg['kwargs'])]
        fc_layers[node] = nn.ModuleList(fc_layers[node])
    fc_layers = nn.ModuleDict(fc_layers)
    return conv_layers, fc_layers


def make_conv_layers(layout, in_channels, out_channels):
    layers = []

    for v in layout:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += conv2d(in_channels, out_channels)
            in_channels = out_channels
    return nn.Sequential(*layers)

def make_fc_layers(in_channels, out_channels, relu=False):
    layers = [nn.Linear(in_channels, out_channels)]
    if relu:
        layers += [nn.ReLU(inplace=True), nn.BatchNorm1d(out_channels)]
    return nn.Sequential(*layers)


def conv2d(in_channels, out_channels):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    layer = [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels)]
    return layer


def get_flattened_dim(layers, img_size):
    x = torch.rand(1,3, *img_size)
    for l in layers:
        x = l[0](x)
    return x.numel()   # flattened_dim

def flatten(x):
    bs = x.size(0)
    return x.view(bs, -1)


class BranchedTiny(nn.Module):
    def __init__(self, ckpt=None, width=64, img_size=(224,224)):
        """Model for multiple facial attribute classification.
        Based on architecture taken as defined in https://arxiv.org/pdf/1904.02920.pdf"""
        super().__init__()
        assert isinstance(img_size, tuple)
        self.w, self.img_size = width, img_size

        self.attributes = ['blurry', 'sideburns', 'wearing_earrings', 'bald', 'goatee', 'mustache',
                           '5_o_clock_shadow', 'arched_eyebrows', 'no_beard', 'heavy_makeup', 'male',
                           'wearing_lipstick', 'black_hair', 'high_cheekbones', 'smiling',
                           'mouth_slightly_open', 'oval_face', 'bushy_eyebrows', 'attractive',
                           'young', 'gray_hair', 'brown_hair', 'blond_hair', 'pale_skin', 'chubby',
                           'double_chin', 'big_nose', 'bags_under_eyes', 'wearing_necklace', 'wearing_necktie',
                           'rosy_cheeks', 'bangs', 'wavy_hair', 'straight_hair', 'wearing_hat', 'big_lips',
                           'narrow_eyes', 'pointy_nose', 'receding_hairline', 'eyeglasses']
        
        self.conv_layers, self.fc_layers = define_net(w=width, img_size=img_size)

        self.register_buffer('celeba_mean', torch.tensor([0.5061, 0.4254, 0.3828]).view(1,3,1,1))
        self.register_buffer('celeba_std', torch.tensor([0.2661, 0.2453, 0.2413]).view(1,3,1,1))

        self.attr_idxs = []

        ckpt = ckpt or 'state.pt'
        self.load(ckpt)
        self.requires_grad_(False)
        self.eval()

    def set_idx_list(self, attributes: list):
        for attr in attributes:
            self.set_idx(attr)

    def set_idx(self, attr: str):
        self.attr_idxs += [self.attributes.index(attr)]

    def normalize(self, imgs):
        return (imgs - self.celeba_mean) / self.celeba_std

    def load(self, ckpt):
        state = torch.load(os.path.join(os.path.dirname(__file__), ckpt), map_location='cpu')
        self.load_state_dict(state)

    def resize(self, x):
        if x.size(-1) != 224:
            x = F.interpolate(x, size=self.img_size, mode='area')
        return x
        
    def forward(self, x):
        x = self.resize(x)
        x = self.normalize(x)

        y6 = self.conv_layers['01-06'][0](x)

        # from_6_to_11 -> 06-11
        y11_0 = self.conv_layers['06-11'][0](y6)
        y11_1 = self.conv_layers['06-11'][1](y6)


        # layers12 -> 11-12

        y12_0 = self.conv_layers['11-12'][0](y11_0)
        y12_1 = self.conv_layers['11-12'][1](y11_0)
        y12_2 = self.conv_layers['11-12'][2](y11_1)

        # layers13 -> self.conv_layers['12-13']

        y13_0 = self.conv_layers['12-13'][0](y12_0)
        y13_1 = self.conv_layers['12-13'][1](y12_0)
        y13_2 = self.conv_layers['12-13'][2](y12_0)
        y13_3 = self.conv_layers['12-13'][3](y12_0)
        y13_4 = self.conv_layers['12-13'][4](y12_1)
        y13_5 = self.conv_layers['12-13'][5](y12_1)
        y13_6 = self.conv_layers['12-13'][6](y12_1)
        y13_7 = self.conv_layers['12-13'][7](y12_1)
        y13_8 = self.conv_layers['12-13'][8](y12_1)
        y13_9 = self.conv_layers['12-13'][9](y12_2)

        y13_0, y13_1, y13_2, y13_3, y13_4, y13_5, y13_6, y13_7, y13_8, y13_9 = \
                map(flatten, (y13_0, y13_1, y13_2, y13_3, y13_4, y13_5, y13_6, y13_7, y13_8, y13_9))


        y14_00 = self.fc_layers['13-14'][0](y13_0)
        y14_01 = self.fc_layers['13-14'][1](y13_1)
        y14_02 = self.fc_layers['13-14'][2](y13_1)
        y14_03 = self.fc_layers['13-14'][3](y13_1)
        y14_04 = self.fc_layers['13-14'][4](y13_2)
        y14_05 = self.fc_layers['13-14'][5](y13_2)
        y14_06 = self.fc_layers['13-14'][6](y13_3)
        y14_07 = self.fc_layers['13-14'][7](y13_4)
        y14_08 = self.fc_layers['13-14'][8](y13_4)
        y14_09 = self.fc_layers['13-14'][9](y13_4)
        y14_10 = self.fc_layers['13-14'][10](y13_5)
        y14_11 = self.fc_layers['13-14'][11](y13_5)
        y14_12 = self.fc_layers['13-14'][12](y13_6)
        y14_13 = self.fc_layers['13-14'][13](y13_6)
        y14_14 = self.fc_layers['13-14'][14](y13_6)
        y14_15 = self.fc_layers['13-14'][15](y13_7)
        y14_16 = self.fc_layers['13-14'][16](y13_8)
        y14_17 = self.fc_layers['13-14'][17](y13_9)

        # Layer 15

        y15_00 = self.fc_layers['14-15'][0](y14_00)
        y15_01 = self.fc_layers['14-15'][1](y14_01)
        y15_02 = self.fc_layers['14-15'][2](y14_02)
        y15_03 = self.fc_layers['14-15'][3](y14_02)
        y15_04 = self.fc_layers['14-15'][4](y14_03)
        y15_05 = self.fc_layers['14-15'][5](y14_03)
        y15_06 = self.fc_layers['14-15'][6](y14_03)
        y15_07 = self.fc_layers['14-15'][7](y14_03)
        y15_08 = self.fc_layers['14-15'][8](y14_04)
        y15_09 = self.fc_layers['14-15'][9](y14_04)
        y15_10 = self.fc_layers['14-15'][10](y14_05)
        y15_11 = self.fc_layers['14-15'][11](y14_06)
        y15_12 = self.fc_layers['14-15'][12](y14_07)
        y15_13 = self.fc_layers['14-15'][13](y14_07)
        y15_14 = self.fc_layers['14-15'][14](y14_07)
        y15_15 = self.fc_layers['14-15'][15](y14_08)
        y15_16 = self.fc_layers['14-15'][16](y14_08)
        y15_17 = self.fc_layers['14-15'][17](y14_09)
        y15_18 = self.fc_layers['14-15'][18](y14_09)
        y15_19 = self.fc_layers['14-15'][19](y14_09)
        y15_20 = self.fc_layers['14-15'][20](y14_10)
        y15_21 = self.fc_layers['14-15'][21](y14_10)
        y15_22 = self.fc_layers['14-15'][22](y14_11)
        y15_23 = self.fc_layers['14-15'][23](y14_11)
        y15_24 = self.fc_layers['14-15'][24](y14_11)
        y15_25 = self.fc_layers['14-15'][25](y14_12)
        y15_26 = self.fc_layers['14-15'][26](y14_13)
        y15_27 = self.fc_layers['14-15'][27](y14_13)
        y15_28 = self.fc_layers['14-15'][28](y14_14)
        y15_29 = self.fc_layers['14-15'][29](y14_15)
        y15_30 = self.fc_layers['14-15'][30](y14_16)
        y15_31 = self.fc_layers['14-15'][31](y14_16)
        y15_32 = self.fc_layers['14-15'][32](y14_17)

        # Last layer

        yf00 = self.fc_layers['15-16'][0](y15_00)
        yf01 = self.fc_layers['15-16'][1](y15_01)
        yf02 = self.fc_layers['15-16'][2](y15_01)
        yf03 = self.fc_layers['15-16'][3](y15_02)
        yf04 = self.fc_layers['15-16'][4](y15_02)
        yf05 = self.fc_layers['15-16'][5](y15_03)
        yf06 = self.fc_layers['15-16'][6](y15_04)
        yf07 = self.fc_layers['15-16'][7](y15_05)
        yf08 = self.fc_layers['15-16'][8](y15_06)
        yf09 = self.fc_layers['15-16'][9](y15_07)
        yf10 = self.fc_layers['15-16'][10](y15_07)
        yf11 = self.fc_layers['15-16'][11](y15_07)
        yf12 = self.fc_layers['15-16'][12](y15_08)
        yf13 = self.fc_layers['15-16'][13](y15_08)
        yf14 = self.fc_layers['15-16'][14](y15_08)
        yf15 = self.fc_layers['15-16'][15](y15_09)
        yf16 = self.fc_layers['15-16'][16](y15_10)
        yf17 = self.fc_layers['15-16'][17](y15_11)
        yf18 = self.fc_layers['15-16'][18](y15_12)
        yf19 = self.fc_layers['15-16'][19](y15_13)
        yf20 = self.fc_layers['15-16'][20](y15_14)
        yf21 = self.fc_layers['15-16'][21](y15_15)
        yf22 = self.fc_layers['15-16'][22](y15_16)
        yf23 = self.fc_layers['15-16'][23](y15_17)
        yf24 = self.fc_layers['15-16'][24](y15_18)
        yf25 = self.fc_layers['15-16'][25](y15_19)
        yf26 = self.fc_layers['15-16'][26](y15_20)
        yf27 = self.fc_layers['15-16'][27](y15_21)
        yf28 = self.fc_layers['15-16'][28](y15_22)
        yf29 = self.fc_layers['15-16'][29](y15_23)
        yf30 = self.fc_layers['15-16'][30](y15_24)
        yf31 = self.fc_layers['15-16'][31](y15_25)
        yf32 = self.fc_layers['15-16'][32](y15_26)
        yf33 = self.fc_layers['15-16'][33](y15_27)
        yf34 = self.fc_layers['15-16'][34](y15_28)
        yf35 = self.fc_layers['15-16'][35](y15_29)
        yf36 = self.fc_layers['15-16'][36](y15_29)
        yf37 = self.fc_layers['15-16'][37](y15_30)
        yf38 = self.fc_layers['15-16'][38](y15_31)
        yf39 = self.fc_layers['15-16'][39](y15_32)

        y = torch.cat((yf00, yf01, yf02, yf03, yf04, yf05, yf06, yf07, yf08, yf09, yf10, yf11, yf12, yf13,
                       yf14, yf15, yf16, yf17, yf18, yf19, yf20, yf21, yf22, yf23, yf24, yf25,
                       yf26, yf27, yf28, yf29, yf30, yf31, yf32, yf33, yf34, yf35, yf36, yf37,
                       yf38, yf39), dim=1)

        if self.attr_idxs:
            y = y[:, self.attr_idxs]

        return y

