import torch
import torch.nn as nn
import gdn_layer
import numpy as np
import math

class res_block(nn.Module):
    def __init__(self,channels):
        super(res_block, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(self,x):
        y = self.conv0(x)
        return x+y

class scale_layer(nn.Module):
    def __init__(self,channel):
        super(scale_layer,self).__init__()
        self.squeeze = nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=True)
        self.flatten = nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=True)

class Hypercoder(nn.Module):
    def __init__(self, N=192, M=320):
        super(Hypercoder, self).__init__()
        num_channels = N
        self.conv_scale = scale_layer(num_channels)

        self.conv_in = nn.Conv2d(M, num_channels, (3, 3), padding=1, bias=True)
        # self.conv_b0 = res_block(channels=num_channels)
        self.prelu0 = nn.PReLU()

        self.conv_b1 = res_block(channels=num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=True)
        self.prelu1 = nn.PReLU()

        self.conv_b2 = res_block(channels=num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(5, 5), stride=(2, 2), padding =2, bias=True)
        self.conv_enc = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1, bias=True)

        self.up0 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=(5, 5), padding=2, output_padding=1, stride=(2, 2))
        self.conv_b3 = res_block(channels=num_channels)
        self.prelu2 = nn.PReLU()

        self.up1 = nn.ConvTranspose2d(num_channels,num_channels, kernel_size=(5, 5), stride=(2, 2),padding=2, output_padding=1)
        self.conv_b4 = res_block(channels=num_channels)
        self.prelu3 = nn.PReLU()

        self.conv_b5 = res_block(channels=num_channels)
        self.conv_loc = nn.Conv2d(num_channels, M, kernel_size=(3, 3), padding=1, bias=True)
        self.conv_scale = nn.Conv2d(num_channels, M, kernel_size=(3, 3), padding=1, bias=True)

    def encoder(self, im, training, scale=False, offset=None):
        #print('hyper_im:', im.shape)
        x1 = self.conv_in(im)  # 3->num_channels
        #print('hyper_conv_in:', x1.shape)
        # x1 = self.conv_b0(x1)
        x2 = self.prelu0(x1)

        x3 = self.conv_b1(x2)
        #print('hyper_conv_b1:', x3.shape)
        x3 = self.conv1(x3)  # downsample
        #print('hyper_conv1:', x3.shape)
        x3 = self.prelu1(x3)

        encoded = self.conv_b2(x3)
        #print('hyper_conv_b2:', encoded.shape)
        encoded = self.conv2(encoded)  # downsample
        #print('hyper_conv2:', encoded.shape)
        # encoded = self.conv_benc(encoded)
        encoded = self.conv_enc(encoded)
        #print('hyper_conv_enc:', encoded.shape)

        if scale == True:
            encoded = self.conv_scale.squeeze(encoded)
            #print('hyper_scale:', encoded.shape)

        # add uniform noise
        if training == True:
            w = torch.Tensor(encoded.shape).cuda()
            dy = nn.init.uniform_(w, a=-0.5, b=0.5)
            encoded = torch.add(encoded, dy)
        else:
            if offset == None:
                encoded = torch.round(encoded)
            else:
                encoded = torch.round(encoded - offset)
        # encoded_cut,c = self.cut(encoded,training = training,c=c)
        return encoded

    def decoder(self, encoded, scale=False):
        if scale == True:
            encoded = self.conv_scale.flatten(encoded)

        x4 = self.up0(encoded)  # conv_transpose
        #('hyper_up0:', x4.shape)
        x4 = self.conv_b3(x4)
        #print('hyper_conv_b3:', x4.shape)
        x4 = self.prelu2(x4)

        x5 = self.up1(x4)  # conv_transpose
        #print('hyper_up1:', x5.shape)
        x5 = self.conv_b4(x5)
        #print('hyper_conv_b4:', x5.shape)
        x6 = self.prelu3(x5)

        x7 = self.conv_b5(x6)
        #print('hyper_conv_b5:', x7.shape)

        loc = self.conv_loc(x7)
        #print('hyper_conv_loc:', loc.shape)
        scale = self.conv_scale(x7)
        #print('hyper_conv_scale:', scale.shape)
        scale[scale.abs() <= 1e-5] = 1e-5
        return loc, scale

    def forward(self, x, training=True, scale=False):
        encoded = self.encoder(x, training, scale=scale)
        loc, scale = self.decoder(encoded, scale=scale)
        return encoded, loc, scale

    def laplace_hyper(self, x, loc, scale):
        c = 0.5 + 0.5 * torch.sign(x - loc) * (1.0 - torch.exp(-torch.abs(x - loc) / scale))  # laplace
        return c

    def cdf_hyper(self, x, loc, scale):
        c = 0.5 * (1.0 + torch.erf((x - loc) / (scale * (2 ** 0.5))))  # gaussian
        return c

class Deepcoder_encoder(nn.Module):
    def __init__(self, N=192, M=320):
        super(Deepcoder_encoder, self).__init__()

        num_channels = N
        self.conv_scale = scale_layer(channel=M)

        self.conv_in = nn.Conv2d(3, num_channels, (3, 3), padding=1, bias=True)
        self.conv0 = nn.Conv2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, bias=True)
        self.gdn0 = gdn_layer.GDN(ch=num_channels,device=0)
        
        self.conv_b1 = res_block(channels=num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, bias=True)
        self.gdn1 = gdn_layer.GDN(ch=num_channels,device=0)
        
        self.conv_b2 = res_block(channels=num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, bias=True)
        self.gdn2 = gdn_layer.GDN(ch=num_channels,device=0)
        
        self.conv_b3 = res_block(channels=num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, bias=True)
        self.conv_benc = res_block(channels=num_channels)
        self.conv_enc = nn.Conv2d(num_channels, M, (3, 3),padding=1, bias=True)


    def forward(self, im, training, scale=False, offset=None):
        #print('im:', im.shape)
        x1 = self.conv_in(im)  # 3->num_channels
        #print('conv_in:', x1.shape)
        x1 = self.conv0(x1)  # downsample
        #print('conv0:', x1.shape)
        x1 = self.gdn0(x1)
        #print('gdn:', x1.shape)
        
        x2 = self.conv_b1(x1)
        #print('conv_b1:', x2.shape)
        x2 = self.conv1(x2)  # downsample
        #print('conv1:', x2.shape)
        x2 = self.gdn1(x2)
        #print('gdn1:', x2.shape)
        
        x3 = self.conv_b2(x2)
        #print('conv_b2:', x3.shape)
        x3 = self.conv2(x3)  # downsample
        #print('conv2:', x3.shape)
        x3 = self.gdn2(x3)
        #print('gdn2:', x3.shape)
        
        encoded = self.conv_b3(x3)
        #print('conv_b3:', encoded.shape)
        encoded = self.conv3(encoded)  # downsample
        #print('conv3:', encoded.shape)
        encoded = self.conv_benc(encoded)
        #print('conv_benc:', encoded.shape)
        encoded_raw = self.conv_enc(encoded)
        #print('encoded_raw:', encoded_raw.shape)

        # TODO: squeeze
        if scale == True:
            encoded_raw = self.conv_scale.squeeze(encoded_raw)
            print('conv_scale:', encoded_raw.shape)

        # add uniform noise
        if training == True:
            w = torch.Tensor(encoded_raw.shape).cuda()
            dy = nn.init.uniform_(w, a=-0.5, b=0.5)
            encoded = torch.add(encoded_raw, dy)
        else:
            if offset == None:
                encoded = torch.round(encoded_raw)
            else:
                encoded = torch.round(encoded_raw - offset)
        # encoded_cut,c = self.cut(encoded,training = training,c=c)
        return encoded_raw, encoded
        
class Deepcoder_decoder(nn.Module):
    def __init__(self, N=192, M=320):
        super(Deepcoder_decoder, self).__init__()

        num_channels = N
        self.up0 = nn.ConvTranspose2d(M, num_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.conv_b4 = res_block(channels=num_channels)
        self.igdn0 = gdn_layer.GDN(ch=num_channels,device=0)
        
        self.up1 = nn.ConvTranspose2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.conv_b5 = res_block(channels=num_channels)
        self.igdn1 = gdn_layer.GDN(ch=num_channels,device=0)
        
        self.up2 = nn.ConvTranspose2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.conv_b6 = res_block(channels=num_channels)
        self.igdn2 = gdn_layer.GDN(ch=num_channels,device=0)
        
        self.up3 = nn.ConvTranspose2d(num_channels, num_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.conv_b7 = res_block(channels=num_channels)
        self.conv_out = nn.Conv2d(num_channels, 3, (3, 3), padding=1, bias=True)


    def forward(self, encoded, scale=False):

        # TODO: flatten
        if scale == True:
            encoded = self.conv_scale.flatten(encoded)
        # encoded = self.conv_scale.flatten(encoded)

        x4 = self.up0(encoded)  # conv_transpose
        #print('up0:', x4.shape)
        x4 = self.conv_b4(x4)
        #print('conv_b4:', x4.shape)
        x4 = self.igdn0(x4)
        #print('igdn0:', x4.shape)
        
        x5 = self.up1(x4)  # conv_transpose
        #print('up1:', x5.shape)
        x5 = self.conv_b5(x5)
        #print('conv_b5:', x5.shape)
        x5 = self.igdn1(x5)
        #print('igdn1:', x5.shape)
        
        x6 = self.up2(x5)  # conv_transpose
        #print('up2:', x6.shape)
        x6 = self.conv_b6(x6)
        #print('conv_b6:', x6.shape)
        x6 = self.igdn2(x6)
        #print('igdn2:', x6.shape)
        
        x7 = self.up3(x6)  # conv_transpose
        #print('up3:', x7.shape)
        x7 = self.conv_b7(x7)
        #print('con_b7:', x7.shape)
        out = self.conv_out(x7)
        #print('conv_out:', out.shape)

        return out

class entropy_estimator(nn.Module):
    def __init__(self, filters_num=3, K=3, channels=192):
        super(entropy_estimator, self).__init__()
        #
        init_scale = 0.5

        filters = [filters_num for x in range(K)]
        self.filters = [1] + filters + [1]

        self.scale = init_scale ** (1.0 / (len(self.filters) + 1.0))

        self.likelihood_bound = 1e-9
        self.K = K
        self.channels = channels
        print(self.filters)
        self.matrices_1 = nn.Parameter(torch.Tensor(self.channels, self.filters[0 + 1], self.filters[0]))
        self.matrices_2 = nn.Parameter(torch.Tensor(self.channels, self.filters[1 + 1], self.filters[1]))
        self.matrices_3 = nn.Parameter(torch.Tensor(self.channels, self.filters[2 + 1], self.filters[2]))
        self.matrices_4 = nn.Parameter(torch.Tensor(self.channels, self.filters[3 + 1], self.filters[3]))

        self.factors_1 = nn.Parameter(torch.Tensor(self.channels, self.filters[0 + 1], 1))
        self.factors_2 = nn.Parameter(torch.Tensor(self.channels, self.filters[1 + 1], 1))
        self.factors_3 = nn.Parameter(torch.Tensor(self.channels, self.filters[2 + 1], 1))
        self.factors_4 = nn.Parameter(torch.Tensor(self.channels, self.filters[3 + 1], 1))

        self.bias_1 = nn.Parameter(torch.Tensor(self.channels, self.filters[0 + 1], 1))
        self.bias_2 = nn.Parameter(torch.Tensor(self.channels, self.filters[1 + 1], 1))
        self.bias_3 = nn.Parameter(torch.Tensor(self.channels, self.filters[2 + 1], 1))
        self.bias_4 = nn.Parameter(torch.Tensor(self.channels, self.filters[3 + 1], 1))


        # Taking the logit (inverse of sigmoid) of the cumulative makes the representation of the right target more numerically stable
        label = torch.log(torch.Tensor([2.0]) / self.likelihood_bound - 1.0)
        label = torch.Tensor([-label, 0.0, label])
        self.label = label.repeat(channels).view(-1,1,3)

        init = torch.Tensor([-10.0, 0.0, 10.0])  # [1,1,3]
        init = init.repeat(channels).view(-1, 1, 3)  # [1,1,3] -> [channels,1,3]

        self.target = init
        self.init_params()

    def init_params(self):
        filters = self.filters
        channels = self.channels
        init_1 = np.log(np.expm1(1.0 / self.scale / filters[0 + 1]))
        init_2 = np.log(np.expm1(1.0 / self.scale / filters[1 + 1]))
        init_3 = np.log(np.expm1(1.0 / self.scale / filters[2 + 1]))
        init_4 = np.log(np.expm1(1.0 / self.scale / filters[3 + 1]))
        self.matrices_1 = nn.init.constant_(self.matrices_1, init_1)
        self.matrices_2 = nn.init.constant_(self.matrices_2, init_2)
        self.matrices_3 = nn.init.constant_(self.matrices_3, init_3)
        self.matrices_4 = nn.init.constant_(self.matrices_4, init_4)

        self.factors_1 = nn.init.constant_(self.factors_1, 0)
        self.factors_2 = nn.init.constant_(self.factors_2, 0)
        self.factors_3 = nn.init.constant_(self.factors_3, 0)
        self.factors_4 = nn.init.constant_(self.factors_4, 0)

        self.bias_1 = nn.init.uniform_(self.bias_1, a=-0.5, b=0.5)
        self.bias_2 = nn.init.uniform_(self.bias_2, a=-0.5, b=0.5)
        self.bias_3 = nn.init.uniform_(self.bias_3, a=-0.5, b=0.5)
        self.bias_4 = nn.init.uniform_(self.bias_4, a=-0.5, b=0.5)


    def _logits_cumulative(self, inputs):  # inputs must have shape [channels,1,N]
        logits = inputs
        #print(i,self.K)
        #print(a.shape)
        logits = torch.matmul(torch.nn.functional.softplus(self.matrices_1), logits) + self.bias_1
        logits = logits + torch.tanh(self.factors_1) * torch.tanh(logits)
        logits = torch.matmul(torch.nn.functional.softplus(self.matrices_2), logits) + self.bias_2
        logits = logits + torch.tanh(self.factors_2) * torch.tanh(logits)
        logits = torch.matmul(torch.nn.functional.softplus(self.matrices_3), logits) + self.bias_3
        logits = logits + torch.tanh(self.factors_3) * torch.tanh(logits)
        logits = torch.matmul(torch.nn.functional.softplus(self.matrices_4), logits) + self.bias_4
        logits = logits + torch.tanh(self.factors_4) * torch.tanh(logits)

        return logits

    def quantile(self):
        logits = self._logits_cumulative(self.target)
        loss = torch.nn.MSELoss(self.label, logits)
        return loss

    def forward(self, x):
        #print(self.matrices_1[0])
        lower = self._logits_cumulative(x - 0.5)
        upper = self._logits_cumulative(x + 0.5)

        # TODO: only compute differences in the left tail of the sigmoid
        sign = - torch.sign(torch.add(lower, upper))
        likelihoods = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        # likelihood bound
        likelihoods[likelihoods<=self.likelihood_bound] = self.likelihood_bound

        return likelihoods

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv0 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1,groups=128,bias = False),
                                   nn.Conv2d(128, 128, 1, 1, 0)
                                  )
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 128,128, layers[0])
        self.layer2 = self._make_layer(block, 128,256, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256,512, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512,1024, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, inplanes,outplanes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes!=outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes,
                          kernel_size=3, stride=stride, padding = 1,bias=True),
                nn.Conv2d(outplanes, outplanes,
                          kernel_size=1, stride= 1, bias=True),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(block(inplanes, outplanes, stride, downsample))
        inplanes = outplanes
        for i in range(1, blocks):
            layers.append(block(inplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

class CodedVision(nn.Module):
    def __init__(self,N,M):
        super (CodedVision,self).__init__()
        
        self.N = N
        self.M = M
        self.Hypercoder = Hypercoder(N=self.N,M=self.M)
        self.Encoder = Deepcoder_encoder(N=self.N,M=self.M)
        self.Decoder = Deepcoder_decoder(N = self.N , M = self.M)
        self.estimator = entropy_estimator(channels=self.N)
        self.classifier = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        
    def forward(self, x, Training=True):
        encoded_raw, encoded = self.Encoder(x, training=Training)
        output = self.Decoder(encoded)
        cls = self.classifier(encoded)
        
        encoded_hyper = self.Hypercoder.encoder(encoded_raw, training=True)
        #print('encoded_hyper:',encoded_hyper.shape)
        loc, scale = self.Hypercoder.decoder(encoded_hyper)
            
        num_pixels = x.size(0)*x.size(2)*x.size(3)            
            
        upper = encoded + 0.5
        lower = encoded - 0.5
        #print(encoded.shape)
        #print(loc.shape)
        sign = (upper + lower - loc).sign()

        upper = - sign * (upper - loc) + loc
        lower = - sign * (lower - loc) + loc

        upper = self.Hypercoder.cdf_hyper(upper, loc, scale)
        lower = self.Hypercoder.cdf_hyper(lower, loc, scale)

        p_laplace = (upper - lower).abs()
        p_laplace[p_laplace <= 1e-6] = 1e-6

        train_bpp_ae = (torch.sum(torch.log(p_laplace))) / -(torch.log(torch.Tensor([2.0]).cuda())) / num_pixels

        encoded_x = encoded_hyper.view(-1,1,128)
        encoded_x = encoded_x.permute(2, 1, 0)
        likelihoods = self.estimator(encoded_x)
        train_bpp_hyper = torch.sum(torch.log(likelihoods)) / -(torch.log(torch.Tensor([2.0]).cuda())) / num_pixels
        bpp = train_bpp_ae + train_bpp_hyper
        return output, bpp,cls,
