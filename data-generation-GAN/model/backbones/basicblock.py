import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, inplanes):
        self.inplanes = inplanes
        super(ResBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=0),
                       nn.BatchNorm2d(self.inplanes),
                       nn.ReLU(True),#nn.LeakyReLU(negative_slope=0.2, inplace=True),#
                       #nn.Dropout(0.5),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=0),
                       nn.BatchNorm2d(self.inplanes),
                       #nn.LeakyReLU(negative_slope=0.2, inplace=True)
                       ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PATBlock(nn.Module):
    def __init__(self, inplanes, order=1):
        super(PATBlock, self).__init__()
        #self.order = 1 if order==1 else 2
        self.conv_block_stream1 = self.build_conv_block(inplanes, order=1)
        self.conv_block_stream2 = self.build_conv_block(inplanes, order=1)

    def build_conv_block(self, inplanes, order=1):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(inplanes*order, inplanes*order, kernel_size=3, padding=0),
                       nn.BatchNorm2d(inplanes*order),
                       nn.ReLU(True),#nn.LeakyReLU(negative_slope=0.2, inplace=True),#
                       #nn.Dropout(0.5),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(inplanes*order, inplanes, kernel_size=3, padding=0),
                       nn.BatchNorm2d(inplanes)
                       #nn.LeakyReLU(negative_slope=0.2, inplace=True)
                       ]

        return nn.Sequential(*conv_block)

    def forward(self, input):
        x1 = input[0]
        x2 = input[1]
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = torch.sigmoid(x2_out)

        x1_out = x1_out * att
        out = x1 + x1_out # residual connection

        # stream2 receive feedback from stream1
        #x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out, x1_out

class PATNs(nn.Module):
    def __init__(self, inplanes, nblocks):
        super(PATNs, self).__init__()
        layers = []
        for i in range(1,nblocks+1):
            layers.append(PATBlock(inplanes, order=i))
        self.layers = nn.Sequential(*layers)
    def forward(self, input):
        x1, x2, _ = self.layers(input)
        # for i, layer in enumerate(self.layers):
        #     x1, x2, _ = layer(input=(x1, x2))
        return x1



class ImageEncoder(nn.Module):
    def __init__(self, nlayers = 2):
        super(ImageEncoder, self).__init__()
        self.inplanes = 64
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, padding=0)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)#nn.LeakyReLU(negative_slope=0.2, inplace=True)#
        self.layers = self._make_layer(nlayers)

    def _make_layer(self, n):
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(self.inplanes*2**i, self.inplanes*2**(i+1), kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm2d(self.inplanes*2**(i+1)),
                       nn.ReLU(inplace=True)#nn.LeakyReLU(negative_slope=0.2, inplace=True)#
                       ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layers(x)
        return x

class PoseEncoder(nn.Module):
    def __init__(self, nlayers = 2):
        super(PoseEncoder, self).__init__()
        self.inplanes = 64
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, padding=0)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)#nn.LeakyReLU(negative_slope=0.2, inplace=True)#
        self.layers = self._make_layer(nlayers)

    def _make_layer(self, n):
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(self.inplanes*2**i, self.inplanes*2**(i+1), kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm2d(self.inplanes*2**(i+1)),
                       nn.ReLU(inplace=True)#nn.LeakyReLU(negative_slope=0.2, inplace=True)#
                       ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layers(x)
        return x


class ImageGenerator(nn.Module):
    def __init__(self, nlayers = 2):
        super(ImageGenerator, self).__init__()
        # up_sample
        self.inplanes = 256
        layers = []
        for i in range(nlayers):
            in_d = 2**i
            out_d = 2**(i+1)
            layers += [nn.ConvTranspose2d(int(self.inplanes/in_d), int(self.inplanes/out_d),
                            kernel_size=3, stride=2,
                            padding=1, output_padding=1),
                       nn.BatchNorm2d(int(self.inplanes/out_d)),
                       nn.ReLU(True)#nn.LeakyReLU(negative_slope=0.2, inplace=True)#
                       ]
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(int(self.inplanes/out_d), 3, kernel_size=7, padding=0),
                   nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x