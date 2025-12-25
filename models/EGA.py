import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import numpy as np
import math

#加入位置编码器

class SinePositionalEncoding2D2(nn.Module):
    def __init__(self, num_feats=64, temperature=10000.0):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.register_buffer('div_term', torch.exp(torch.arange(0, num_feats, 2) *
                                                   -(math.log(temperature) / num_feats)))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)
        div_term = self.div_term.to(device)
        pos_x = x_embed.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0)
        pos_y = y_embed.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0)
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        if pos.size(1) > C:
            pos = pos[:, :C, :, :]
        elif pos.size(1) < C:
            pos = torch.cat([pos, torch.zeros(1, C - pos.size(1), H, W, device=device)], dim=1)
        return x + pos.repeat(B, 1, 1, 1)

class ConvPositionalEncoding2D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.pe_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        return x + self.pe_conv(x)

class _ChannelGate(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        hidden = max(ch // r, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class _SpatialGate(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.proj = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.sigmoid(self.proj(torch.cat([avg, mx], dim=1)))
        return s

class DGA(nn.Module):
    def __init__(
        self,
        c_channels: int,
        e_channels: int,
        level: str = "low",
        reduction: int = 16,
        ds_kernel: int = 3,
        use_depthwise: bool = True,
        conf_keep: float = 0.5,
        conf_drop: float = 0.1,
        detach_gates: bool = False,
        high_alpha_cap: float = 0.0,
    ):
        super().__init__()
        self.level = level.lower()
        self.conf_keep = float(conf_keep)
        self.conf_drop = float(conf_drop)
        self.detach_gates = bool(detach_gates)
        self.high_alpha_cap = float(high_alpha_cap)

        self.adapter_e2c = nn.Conv2d(e_channels, c_channels, 1, bias=False)

        self.edge_conf = nn.Sequential(
            nn.Conv2d(e_channels, max(e_channels // 4, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(e_channels // 4, 8), 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.c_gate_ch = _ChannelGate(c_channels, r=reduction)
        self.c_gate_sp = _SpatialGate(k=ds_kernel)

        self.e_gate_ch = _ChannelGate(e_channels, r=reduction)
        self.e_gate_sp = _SpatialGate(k=ds_kernel)

        if use_depthwise:
            self.ds = nn.Sequential(
                nn.Conv2d(c_channels, c_channels, ds_kernel, padding=ds_kernel // 2, groups=c_channels, bias=False),
                nn.Conv2d(c_channels, c_channels, 1, bias=False),
            )
        else:
            self.ds = nn.Identity()

        self.mix_lambda = nn.Parameter(torch.tensor(0.5))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

        self.alpha = nn.Parameter(torch.zeros(1))
        self.alpha_pixel = nn.Sequential(
            nn.Conv2d(c_channels * 2, max(c_channels // 4, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c_channels // 4, 8), 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.alpha_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_channels, max(c_channels // 16, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c_channels // 16, 4), c_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 初始化
        nn.init.kaiming_normal_(self.adapter_e2c.weight, mode="fan_out", nonlinearity="relu")
        if isinstance(self.ds, nn.Sequential):
            for m in self.ds.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if self.level == "low":
            with torch.no_grad():
                self.alpha.copy_(torch.tensor(1.0))
        elif self.level == "high":
            with torch.no_grad():
                self.alpha.copy_(torch.tensor(-2.0))

    def forward(self, c_feat: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        if e_feat.shape[2:] != c_feat.shape[2:]:
            e_feat = F.interpolate(e_feat, size=c_feat.shape[2:], mode="bilinear", align_corners=True)
        e_clean = e_feat * self.e_gate_ch(e_feat) * self.e_gate_sp(e_feat)

        conf = self.edge_conf(e_feat)
        if self.training and self.conf_drop > 0.0:
            conf = F.dropout(conf, p=self.conf_drop, training=True)
        conf = self.conf_keep + (1.0 - self.conf_keep) * conf
        e_clean = e_clean * conf

        e_proj = self.adapter_e2c(e_clean)

        c_in_gate = c_feat.detach() if self.detach_gates else c_feat
        gate = self.mix_lambda * self.c_gate_ch(c_in_gate) + (1 - self.mix_lambda) * self.c_gate_sp(c_in_gate)
        gate = torch.clamp(gate + self.gate_bias, 0.0, 1.0)

        gated_e = e_proj * gate

        delta = self.ds(gated_e)

        alpha_s = torch.sigmoid(self.alpha)
        alpha_p = self.alpha_pixel(torch.cat([c_feat, gated_e], dim=1))
        alpha_c = self.alpha_channel(c_feat)
        alpha_eff = 0.5 * alpha_s + 0.35 * alpha_p + 0.15 * alpha_c
        alpha_eff = torch.clamp(alpha_eff, 0.0, 1.0)

        if self.level == "high" and self.high_alpha_cap > 0.0:
            alpha_eff = torch.clamp(alpha_eff, 0.0, self.high_alpha_cap)

        return c_feat + alpha_eff * delta



def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class EdgeAttentionModule(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(EdgeAttentionModule, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return out


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_input=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(pretrained=False, layers=[3, 4, 6, 3], backbone='resnet50', n_input=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, layers, n_input=n_input, **kwargs)

    pretrain_dict = model_zoo.load_url(model_urls[backbone])
    try:
        model.load_state_dict(pretrain_dict, strict=False)
    except:
        print("loss conv1")
        model_dict = {}
        for k, v in pretrain_dict.items():
            if k in pretrain_dict and 'conv1' not in k:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
    print("load pretrain success")
    return model


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, n_input=3):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
        self.relu = self.model.relu

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out

class ChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 5, 7, 9, 11):
            raise ValueError("kernel_size must be an odd number, e.g., 7")
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        return self.proj(att)
class EAEB(nn.Module):
    def __init__(self, in_channels, out_channels, reduction= 16, kernel_size= 7):
        super(EAEB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.channel_att = ChannelAttention(out_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)


    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        res = res * self.channel_att(res)
        res = res * self.spatial_att(res)

        if relu:
            return self.relu(x + res)
        else:
            return x + res


class BasicBlock_ups(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_ups, self).__init__()

        self.ups = nn.Sequential()
        if stride == 2:
            self.ups = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.ups(out)
        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=3):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EGANet(ResNet50):
    def __init__(self, nclass, BatchNorm=nn.BatchNorm2d, aux=False, sobel=False, constrain=False, n_input=3, **kwargs):
        super(EGANet, self).__init__(pretrained=True, n_input=n_input)
        self.num_class = nclass
        self.aux = aux

        self.sinpe = SinePositionalEncoding2D2(2048)
        self.convpe = ConvPositionalEncoding2D(256)

        self.cross1 = DGA(c_channels=256, e_channels=32, level="low")
        self.cross2 = DGA(512, 64, level="low", high_alpha_cap=0.25)
        self.cross3 = DGA(1024, 128,level="high", detach_gates=True, high_alpha_cap=0.2)
        self.cross4 = DGA(2048, 256, level="high", detach_gates=True, high_alpha_cap=0.2)

        self.__setattr__('exclusive', ['head'])
        self.BatchNorm = BatchNorm
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.sobel = sobel
        self.constrain = constrain

        self.erb_db_1 = EAEB(256, 256)
        self.erb_db_2 = EAEB(512, 256)
        self.erb_db_3 = EAEB(1024, 256)
        self.erb_db_4 = EAEB(2048, 256)
        if self.sobel:
            print("----------use sobel-------------")
            self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(1024, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(2048, 1)


        self.head = GARHead(2048 + 2048, self.num_class, aux)


        self.edge_attention = EdgeAttentionModule(2048, 256, 256, BatchNorm)
        self.edge_down = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.myin_planes = 3
        self.mylayer1 = self._make_layer(BasicBlock2, 32, 1, stride=2, kernel_size=3)
        self.mylayer2 = self._make_layer(BasicBlock2, 64, 1, stride=2, kernel_size=3)
        self.mylayer3 = self._make_layer(BasicBlock2, 128, 2, stride=2, kernel_size=3)
        self.mylayer4 = self._make_layer(BasicBlock2, 256, 2, stride=2, kernel_size=3)

        self.myUpsamplelayer1 = self._make_layer_ups(BasicBlock_ups, 128, 2,
                                                     stride=2)
        self.myUpsamplelayer2 = self._make_layer_ups(BasicBlock_ups, 64, 1,
                                                     stride=2)

        self.myconv_layer = nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1)
        self.mymodel = resnet(n_input=n_input, layers=[3, 4, 6, 3], backbone='resnet50')
        mylayers_cfg = [4, 5, 6, 7]
        self.myblocks = []
        for i, num_this_layer in enumerate(mylayers_cfg):
            self.myblocks.append(list(self.mymodel.children())[num_this_layer])

    def _make_layer_2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.myin_planes, planes, stride, kernel_size))
            self.myin_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def _make_layer_ups(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.myin_planes, planes, stride))
            self.myin_planes = planes * block.expansion
        return nn.Sequential(*layers)

    #################################################################
    def forward(self, x):

        size = x.size()[2:]
        input_ = x.clone()

        out1 = self.mylayer1(x)
        out2 = self.mylayer2(out1)
        out3 = self.mylayer3(out2)
        out4 = self.mylayer4(out3)
        feature_map, _ = self.base_forward(input_)
        c1, c2, c3, c4 = feature_map
        c1 = self.convpe(c1)
        c4 = self.sinpe(c4)
        c1 = self.cross1(c1, out1)
        c4 = self.cross4(c4, out4)
        out33 = self.myUpsamplelayer1(out4)
        out22 = self.myUpsamplelayer2(out33)
        myout = torch.abs(out22 - out2)
        my_feature_map = []
        myret2 = torch.mean(myout)
        for i, block in enumerate(self.myblocks):
            if i == 0:
                myx = block(myout)
            else:
                myx = block(myx)
            my_feature_map.append(myx)

        mya1, mya2, mya3, mya4 = my_feature_map

        if self.sobel:
            res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
            res2 = self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))
            res2 = F.interpolate(res2, scale_factor=2, mode='bilinear', align_corners=True)
            res3 = self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))
            res3 = F.interpolate(res3, scale_factor=4, mode='bilinear', align_corners=True)
            res4 = self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))
            res4 = F.interpolate(res4, scale_factor=4, mode='bilinear', align_corners=True)

        else:
            res1 = self.erb_db_1(c1)
            res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
            res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
            res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)
        E = torch.cat((res1, res2, res3, res4), dim=1)
        E = self.edge_down(E)

        out_feature = self.edge_attention(c4, E)


        # Final output
        edge = self.myconv_layer(out_feature)
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=True)

        h4 = torch.cat([mya4, c4], dim=1)

        head_outputs = self.head(h4)


        outputs = []
        if isinstance(head_outputs, tuple):
            for out in head_outputs:
                outputs.append(F.interpolate(out, size, mode='bilinear', align_corners=True))
        else:
            outputs.append(F.interpolate(head_outputs, size, mode='bilinear', align_corners=True))

        return edge, outputs[0], myret2

class GChannelAttention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super().__init__()
        hidden = max(int(in_channels / rate), 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)  # [B,HW,C]
        att = self.mlp(x_perm).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        att = torch.sigmoid(att)
        return x * att


class GSpatialAttention(nn.Module):

    def __init__(self, in_channels, rate=4, kernel_size=7):
        super().__init__()
        hidden = max(int(in_channels / rate), 1)
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=pad, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=kernel_size, padding=pad, bias=True),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        att = torch.sigmoid(self.net(x))
        return x * att

class GARHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, reduction=8):
        super(GARHead, self).__init__()
        self.aux = aux
        inter_channels = max(in_channels // reduction, 64)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # 这里就是“用空间和通道类替代 GAM”的实现
        self.gam_ca = GChannelAttention(inter_channels, rate=4)
        self.gam_sa = GSpatialAttention(inter_channels, rate=4, kernel_size=7)

        self.main_conv = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )

        if aux:
            self.aux_conv = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels // 2, 1, bias=False),
                nn.BatchNorm2d(inter_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels // 2, nclass, 1)
            )

    def forward(self, x):
        feat = self.conv_in(x)
        feat = self.gam_ca(feat)
        feat = self.gam_sa(feat)
        main_out = self.main_conv(feat)
        return main_out


def get_ega(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, n_input=3, constrain=False, **kwargs):
    model = EGANet(nclass, backbone=backbone,
                    pretrained_base=pretrained_base,
                    sobel=sobel,
                    n_input=n_input,
                    constrain=constrain,
                    **kwargs)
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 512, 512)
    model = get_ega(sobel=True, n_input=3, pretrained_base=True, constrain=True)
    edge, outputs, ret2 = model(img)
    print(outputs.shape)
    print(edge.shape)