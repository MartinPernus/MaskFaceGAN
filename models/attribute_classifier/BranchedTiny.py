
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class BranchedTinyAttr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = BranchedTiny(cfg.CKPT)
        self.img_size = 224
        self.interpolate_cfg = {'size': (self.img_size, self.img_size), 'mode': 'area'}
        self.idx = []

    def set_idx_list(self, attributes):
        for attr in attributes:
            self.set_idx(attr)

    def set_idx(self, attr):
        self.idx += [self.model.attributes.index(attr)]

    def forward(self, img):
        if img.size(-1) != self.img_size:
            img = F.interpolate(img, **self.interpolate_cfg)
        y = self.model(img)
        if self.idx is not None:
            y = y[:, self.idx]
        return y

class BranchedTiny(nn.Module):
    def __init__(self, ckpt=None, width=64, img_size=(224,224)):
        """From paper https://arxiv.org/pdf/1904.02920.pdf"""
        super().__init__()
        assert isinstance(img_size, tuple)
        self.w, self.img_size= width, img_size

        self.attributes = ['blurry', 'sideburns', 'wearing_earrings', 'bald', 'goatee', 'mustache',
                           '5_o_clock_shadow', 'arched_eyebrows', 'no_beard', 'heavy_makeup', 'male',
                           'wearing_lipstick', 'black_hair', 'high_cheekbones', 'smiling',
                           'mouth_slightly_open', 'oval_face', 'bushy_eyebrows', 'attractive',
                           'young', 'gray_hair', 'brown_hair', 'blond_hair', 'pale_skin', 'chubby',
                           'double_chin', 'big_nose', 'bags_under_eyes', 'wearing_necklace', 'wearing_necktie',
                           'rosy_cheeks', 'bangs', 'wavy_hair', 'straight_hair', 'wearing_hat', 'big_lips',
                           'narrow_eyes', 'pointy_nose', 'receding_hairline', 'eyeglasses']

        total_cfg = [self.w, self.w, 'M', self.w, self.w, 'M',  self.w, self.w,  # first 6 layers
                     self.w, 'M', self.w, self.w, self.w, 'M', self.w,
                     self.w, self.w, 'M']

        first_6_cfg = total_cfg[:8]
        _7_to_11_cfg = total_cfg[8:15]
        _12_to_13_cfg = total_cfg[15:16]
        _13_to_14_cfg = total_cfg[16:18]

        self.to_layer_6 = self.make_conv_layers(first_6_cfg, in_channels=3)
        self.n_layers = {'layers7_to_11': 2, 'layers12': 3, 'layers13': 10,
                         'layers14': 18, 'layers15': 33, 'layers16': 40}

        from_6_to_11, layers12, layers13, layers14, layers15, layers16 = [], [], [], [], [], []
        for _ in range(self.n_layers['layers7_to_11']): from_6_to_11.append(
            self.make_conv_layers(_7_to_11_cfg, self.w))

        for _ in range(self.n_layers['layers12']): layers12.append(self.make_conv_layers(_12_to_13_cfg, self.w))
        for _ in range(self.n_layers['layers13']): layers13.append(self.make_conv_layers(_13_to_14_cfg, self.w))

        # Model construction for fully-connected models

        self.flattened_dim = layers13[0](layers12[0](from_6_to_11[0](
            self.to_layer_6(torch.rand(1, 3, *img_size))))).numel()
        for _ in range(self.n_layers['layers14']): layers14.append(self.fc(self.flattened_dim, 2*self.w, ReLU=True))
        for _ in range(self.n_layers['layers15']): layers15.append(self.fc(2*self.w, 2*self.w, ReLU=True))
        for _ in range(self.n_layers['layers16']): layers16.append(self.fc(2*self.w, 1, ReLU=False))

        # Create ModuleLists out of the branches
        self.from_6_to_11, self.layers12, self.layers13, self.layers14, self.layers15, self.layers16 = \
            map(lambda x: nn.ModuleList(x), (
                from_6_to_11, layers12, layers13, layers14, layers15, layers16
            ))

        self.celeba_aligned_means= nn.Parameter(torch.tensor([0.5061, 0.4254, 0.3828]), requires_grad=False)
        self.celeba_aligned_stds = nn.Parameter(torch.tensor([0.2661, 0.2453, 0.2413]), requires_grad=False)

        if ckpt is not None:
            self.load(ckpt)

    def __normalize(self, imgs):
        return (imgs - self.celeba_aligned_means[None, :, None, None]) / self.celeba_aligned_stds[None, :, None, None]

    def load(self, ckpt):
        path = os.path.join(os.path.dirname(__file__), ckpt)
        state_dict = torch.load(path, map_location=torch.device('cuda:0'))['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if 'model.' in k:
                newk = k[6:]
            else:
                newk = k
            new_state_dict[newk] = v

        self.load_state_dict(new_state_dict)

    def forward(self, x):
        if x.size(-1) != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear')
        x = self.__normalize(x)
        BS = x.size(0)
        y6 = self.to_layer_6(x)

        # Up to Layer 11

        y11_0 = self.from_6_to_11[0](y6)
        y11_1 = self.from_6_to_11[1](y6)

        # Layer 12

        y12_0 = self.layers12[0](y11_0)
        y12_1 = self.layers12[1](y11_0)
        y12_2 = self.layers12[2](y11_1)

        # Layer 13

        y13_0 = self.layers13[0](y12_0)
        y13_1 = self.layers13[1](y12_0)
        y13_2 = self.layers13[2](y12_0)
        y13_3 = self.layers13[3](y12_0)
        y13_4 = self.layers13[4](y12_1)
        y13_5 = self.layers13[5](y12_1)
        y13_6 = self.layers13[6](y12_1)
        y13_7 = self.layers13[7](y12_1)
        y13_8 = self.layers13[8](y12_1)
        y13_9 = self.layers13[9](y12_2)

        y13_0, y13_1, y13_2, y13_3, y13_4, y13_5, y13_6, y13_7, y13_8, y13_9 = \
                map(lambda x: x.view(BS, -1), (y13_0, y13_1, y13_2, y13_3, y13_4, y13_5,
                                           y13_6, y13_7, y13_8, y13_9))


        y14_0 = self.layers14[0](y13_0)
        y14_1 = self.layers14[1](y13_1)
        y14_2 = self.layers14[2](y13_1)
        y14_3 = self.layers14[3](y13_1)
        y14_4 = self.layers14[4](y13_2)
        y14_5 = self.layers14[5](y13_2)
        y14_6 = self.layers14[6](y13_3)
        y14_7 = self.layers14[7](y13_4)
        y14_8 = self.layers14[8](y13_4)
        y14_9 = self.layers14[9](y13_4)
        y14_10 = self.layers14[10](y13_5)
        y14_11 = self.layers14[11](y13_5)
        y14_12 = self.layers14[12](y13_6)
        y14_13 = self.layers14[13](y13_6)
        y14_14 = self.layers14[14](y13_6)
        y14_15 = self.layers14[15](y13_7)
        y14_16 = self.layers14[16](y13_8)
        y14_17 = self.layers14[17](y13_9)

        # Layer 15

        y15_0 = self.layers15[0](y14_0)
        y15_1 = self.layers15[1](y14_1)
        y15_2 = self.layers15[2](y14_2)
        y15_3 = self.layers15[3](y14_2)
        y15_4 = self.layers15[4](y14_3)
        y15_5 = self.layers15[5](y14_3)
        y15_6 = self.layers15[6](y14_3)
        y15_7 = self.layers15[7](y14_3)
        y15_8 = self.layers15[8](y14_4)
        y15_9 = self.layers15[9](y14_4)
        y15_10= self.layers15[10](y14_5)
        y15_11= self.layers15[11](y14_6)
        y15_12= self.layers15[12](y14_7)
        y15_13= self.layers15[13](y14_7)
        y15_14= self.layers15[14](y14_7)
        y15_15= self.layers15[15](y14_8)
        y15_16= self.layers15[16](y14_8)
        y15_17= self.layers15[17](y14_9)
        y15_18= self.layers15[18](y14_9)
        y15_19= self.layers15[19](y14_9)
        y15_20= self.layers15[20](y14_10)
        y15_21= self.layers15[21](y14_10)
        y15_22= self.layers15[22](y14_11)
        y15_23= self.layers15[23](y14_11)
        y15_24= self.layers15[24](y14_11)
        y15_25= self.layers15[25](y14_12)
        y15_26= self.layers15[26](y14_13)
        y15_27= self.layers15[27](y14_13)
        y15_28= self.layers15[28](y14_14)
        y15_29= self.layers15[29](y14_15)
        y15_30= self.layers15[30](y14_16)
        y15_31= self.layers15[31](y14_16)
        y15_32= self.layers15[32](y14_17)

        # Last layer

        yf0 = self.layers16[0](y15_0)
        yf1 = self.layers16[1](y15_1)
        yf2 = self.layers16[2](y15_1)
        yf3 = self.layers16[3](y15_2)
        yf4 = self.layers16[4](y15_2)
        yf5 = self.layers16[5](y15_3)
        yf6 = self.layers16[6](y15_4)
        yf7 = self.layers16[7](y15_5)
        yf8 = self.layers16[8](y15_6)
        yf9 = self.layers16[9](y15_7)
        yf10= self.layers16[10](y15_7)
        yf11= self.layers16[11](y15_7)
        yf12= self.layers16[12](y15_8)
        yf13= self.layers16[13](y15_8)
        yf14= self.layers16[14](y15_8)
        yf15= self.layers16[15](y15_9)
        yf16= self.layers16[16](y15_10)
        yf17= self.layers16[17](y15_11)
        yf18= self.layers16[18](y15_12)
        yf19= self.layers16[19](y15_13)
        yf20= self.layers16[20](y15_14)
        yf21= self.layers16[21](y15_15)
        yf22 = self.layers16[22](y15_16)
        yf23 = self.layers16[23](y15_17)
        yf24 = self.layers16[24](y15_18)
        yf25 = self.layers16[25](y15_19)
        yf26 = self.layers16[26](y15_20)
        yf27 = self.layers16[27](y15_21)
        yf28 = self.layers16[28](y15_22)
        yf29 = self.layers16[29](y15_23)
        yf30 = self.layers16[30](y15_24)
        yf31 = self.layers16[31](y15_25)
        yf32 = self.layers16[32](y15_26)
        yf33 = self.layers16[33](y15_27)
        yf34 = self.layers16[34](y15_28)
        yf35 = self.layers16[35](y15_29)
        yf36 = self.layers16[36](y15_29)
        yf37 = self.layers16[37](y15_30)
        yf38 = self.layers16[38](y15_31)
        yf39 = self.layers16[39](y15_32)

        y = torch.cat((yf0, yf1, yf2, yf3, yf4, yf5, yf6, yf7, yf8, yf9, yf10, yf11, yf12, yf13,
                       yf14, yf15, yf16, yf17, yf18, yf19, yf20, yf21, yf22, yf23, yf24, yf25,
                       yf26, yf27, yf28, yf29, yf30, yf31, yf32, yf33, yf34, yf35, yf36, yf37,
                       yf38, yf39), dim=1)

        return y

    def make_conv_layers(self, cfg, in_channels):
        layers = []

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += self.conv2d(in_channels, self.w)
                in_channels = self.w
        return nn.Sequential(*layers)

    def fc(self, in_channels, out_channels, ReLU=False):
        layers = [nn.Linear(in_channels, out_channels)]
        if ReLU:
            layers += [nn.ReLU(inplace=True), nn.BatchNorm1d(out_channels)]
        return nn.Sequential(*layers)


    def conv2d(self, in_channels, out_channels):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        layer = [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels)]
        return layer
