import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from thop import clever_format, profile
from models._blocks import Conv3x3
from models.backbones.resnet import resnet18
from models.contour import Contour_Exectraction_Module, Contour_Exectraction_Module2
from copy import deepcopy
from models.cross_transformer_encoder import Cross_Transformer_Encoder, TransformerDecoder



class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):  # in:32,out: 2
        super().__init__(
            Conv3x3(in_ch, in_ch, norm=True, act=True),
            Conv3x3(in_ch, out_ch)
        )

class JointAtt(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels // 2, nn.BatchNorm2d, kernel_size=1, padding=0))
        self.key1 = nn.Sequential(BasicConv2d(in_channels, in_channels, nn.BatchNorm2d, kernel_size=1, padding=0))
        self.value1 = nn.Sequential(BasicConv2d(in_channels, in_channels, nn.BatchNorm2d, kernel_size=1, padding=0))

        self.query2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels // 2, nn.BatchNorm2d, kernel_size=1, padding=0))
        self.key2 = nn.Sequential(BasicConv2d(in_channels, in_channels, nn.BatchNorm2d, kernel_size=1, padding=0))
        self.value2 = nn.Sequential(BasicConv2d(in_channels, in_channels, nn.BatchNorm2d, kernel_size=1, padding=0))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input1, input2):
        q1 = self.query1(input1.unsqueeze(3)).squeeze(3)
        k1 = self.key1(input1.unsqueeze(3)).squeeze(3)
        v1 = self.value1(input1.unsqueeze(3)).squeeze(3)

        q2 = self.query2(input2.unsqueeze(3)).squeeze(3)
        k2 = self.key2(input2.unsqueeze(3)).squeeze(3)
        v2 = self.value2(input2.unsqueeze(3)).squeeze(3)

        q = torch.cat([q1, q2], 1).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1 + input1

        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2 + input2

        return out1, out2  # 8 x 256,16,16


class CustomBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super(CustomBlock, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Feature_Pyramid_decoder(nn.Module):
    def __init__(self, out_dim, BatchNorm):
        super(Feature_Pyramid_decoder, self).__init__()
        self.out_dim = out_dim  # 32
        self.bran1 = CustomBlock(64, 96)
        self.bran2 = CustomBlock(128, 96)
        self.bran3 = CustomBlock(256, 96)
        self.Conv = nn.Sequential(
            nn.Conv2d(288, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.last_conv = nn.Sequential(nn.Conv2d(258, self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.out_dim),
                                       nn.ReLU())

    def forward(self, x1_3, x1_2, x1_1, contour):
        x3 = self.bran3(x1_3)
        x2 = self.bran2(x1_2)
        x1 = self.bran1(x1_1)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = self.Conv(torch.cat((x3, x2, x1), dim=1))
        x_fuse_edge = torch.cat((x, contour), dim=1)
        out = self.last_conv(x_fuse_edge)
        return out


def build_decoder(out_d, BatchNorm):
    return Feature_Pyramid_decoder(out_d, BatchNorm)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):  # 这个是特征缩减的比例
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 中间全连接层
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 和输入通道不变
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class GCN(nn.Module):
    def __init__(self, dim, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # bs c k
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)  # bs k c x weight(K x K) = bs k c --> bs c k
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1. / math.sqrt(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class CGIModule(nn.Module):
    def __init__(self, dim_f, dim_vertex, mids, BatchNorm=nn.BatchNorm2d, normalize=False, dropout=0.1):
        super(CGIModule, self).__init__()

        self.normalize = normalize
        self.dim = int(dim_vertex)  # 中间的节点数量
        self.num_vertex = (mids) * (mids)
        self.dim_vertex = dim_vertex
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(dim_f, self.dim, kernel_size=1)
        self.conv_proj = nn.Conv2d(dim_f, self.dim, kernel_size=1)
        self.conv_extend = nn.Conv2d(self.dim, dim_f, kernel_size=1, bias=False)
        self.gcn = GCN(dim=self.dim, num_node=self.num_vertex)
        self.joint = JointAtt(self.dim_vertex)
        self.conv_extend = nn.Conv2d(self.dim, dim_f, kernel_size=1, bias=False)
        self.blocker = BatchNorm(dim_f)

    def _reshape_and_softmax(self, x, contour):
        n, c, h, w = x.size()
        contour = F.upsample(contour, (h, w))
        contour = torch.nn.functional.softmax(contour, dim=1)[:, 1, :, :].unsqueeze(1)
        x_proj = self.conv_proj(x)  # bs num_k:128 16 16
        x_mask = x_proj * contour  # bs num_k:128 16 16
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.dim, -1)  # bs c num_vertex
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1),
                                       x_proj.reshape(n, self.dim, -1))  # num_vertex N(hxw)
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)  # bs num_vertex N

        return x_proj_reshaped

    def forward(self, x1, x2, contour1, contour2):
        n, c, h, w = x1.size()
        # 获得投影矩阵
        x_proj_reshaped1 = self._reshape_and_softmax(x1, contour1)  # num_vertex N
        x_proj_reshaped2 = self._reshape_and_softmax(x2, contour2)
        # projection成图节点, # bs c N   x  bs N num_vertex
        x1_g = torch.matmul(self.conv_state(x1).view(x1.size(0), self.dim, -1),
                            x_proj_reshaped1.permute(0, 2, 1))  # bs c num_vertex
        x2_g = torch.matmul(self.conv_state(x2).view(x2.size(0), self.dim, -1),
                            x_proj_reshaped2.permute(0, 2, 1))  # bs c num_vertex

        if self.normalize:
            x1_g = x1_g * (1. / x1.size(-1))  # 2 C k
            x2_g = x2_g * (1. / x2.size(-1))

        # GIM
        x1_g, x2_g = self.joint(x1_g, x2_g)
        x1_g = self.gcn(x1_g)
        x2_g = self.gcn(x2_g)

        # 投影回特征图
        x1_f = torch.matmul(x1_g, x_proj_reshaped1)
        x2_f = torch.matmul(x2_g, x_proj_reshaped2)

        x1_f = x1_f.view(n, self.dim, *x1.size()[2:])
        x2_f = x2_f.view(n, self.dim, *x2.size()[2:])
        out1 = x1 + self.blocker(self.conv_extend(x1_f))
        out2 = x2 + self.blocker(self.conv_extend(x2_f))
        return out1, out2


class CTT_CGINet(nn.Module):
    def __init__(
            self, in_ch=3, out_ch=2, pretrain=False, ckpt_path=None, de_out=32, ratio=8, kernel=7,
            token_len=4
    ):
        super().__init__()

        self.pretrain = pretrain  # 是否采用seasona-contrast的遥感预训练权重
        self.token_len = token_len  # layer4的特征聚合成token的数目
        if self.pretrain:
            model = MocoV2.load_from_checkpoint(ckpt_path)
            self.resnet = deepcopy(model.encoder_q)
        else:
            self.resnet = resnet18(pretrained=True, replace_stride_with_dilation=[False, False, True])
            self.resnet.layer4 = nn.Identity()
            self.resnet.avgpool = nn.Identity()
            self.resnet.fc = nn.Identity()

        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear')  # 用于对差分图像进行线性插值上采样，（64，,64） -》 （256,256）
        self.transform_input = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.donwsample32 = nn.Conv2d(256, 32, kernel_size=1)
        self.cross_transformer_encoder = Cross_Transformer_Encoder(dim=32, depth=1, heads=8,
                                                                   dim_head=64,
                                                                   mlp_dim=64, dropout=0)
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.transformer_decoder = TransformerDecoder(dim=32, depth=8,
                                                      heads=8, dim_head=64, mlp_dim=64,
                                                      dropout=0,
                                                      softmax=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, 32))
        self.classifier = DoubleConv(96, out_ch)  # diff concat经过classfier
        self.MGLNet3 = CGIModule(256, 128, 4)  # 16x16 投影成16图节点
        self.MGLNet2 = CGIModule(128, 64, 6)  # 32X32 投影成36图节点
        self.MGLNet1 = CGIModule(64, 64, 8)  # 64x64 投影成64图节点
        self.cbam0 = CBAM(64, ratio, kernel)
        self.cbam1 = CBAM(64, ratio, kernel)
        self.decoder = build_decoder(64, nn.BatchNorm2d)
        self.cem = Contour_Exectraction_Module()

    def get_token(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)  # 32 64 64 --> num_token 64 64
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()  # 1,num_token,(HXW)
        spatial_attention = torch.softmax(spatial_attention, dim=-1)  # 转为0,1概率
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)  # l c

        return tokens

    def cross_transformer(self, x1, x2):
        x1 += self.pos_embedding
        x2 += self.pos_embedding
        x = self.transformer(x1, x2)  # 生成T1,2;T2,1
        return x

    def cross_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, t1, t2):
        # 主干特征提取网络
        if self.pretrain is False:
            c0 = self.resnet.conv1(t1)  # 1,3,256,256 --> 1,64,128,128
            c0 = self.resnet.bn1(c0)
            c0 = self.resnet.relu(c0)
            c1 = self.resnet.maxpool(c0)  # 1x64,64,64
            c1 = self.resnet.layer1(c1)  # 1,64,64,64
            c2 = self.resnet.layer2(c1)  # 1,128,32,32
            c3 = self.resnet.layer3(c2)  # 1,256,16,16
            c4 = self.resnet.layer4(c3)
            c4_up4x = self.upsample4x(c4)
            tr_in = self.transform_input(c4_up4x)

            c0_img2 = self.resnet.conv1(t2)
            c0_img2 = self.resnet.bn1(c0_img2)
            c0_img2 = self.resnet.relu(c0_img2)
            c1_img2 = self.resnet.maxpool(c0_img2)
            c1_img2 = self.resnet.layer1(c1_img2)
            c2_img2 = self.resnet.layer2(c1_img2)
            c3_img2 = self.resnet.layer3(c2_img2)
            c4_img2 = self.resnet.layer4(c3_img2)
            c4_img2_up4x = self.upsample4x(c4_img2)
            tr_in_img2 = self.transform_input(c4_img2_up4x)

        '''Graph 分支'''
        contour1 = self.cem(c1, c2, c3)  # 使用layer1,layer2,layer3进行轮廓图生产
        contour2 = self.cem(c1_img2, c2_img2, c3_img2)
        x1_3, x2_3 = self.MGLNet3(c3, c3_img2, contour1, contour2)  # bs 128 16 16
        x1_2, x2_2 = self.MGLNet2(c2, c2_img2, contour1, contour2)  # bs 64 32 32
        x1_1, x2_1 = self.MGLNet1(c1, c1_img2, contour1, contour2)  # bs 64 64 64
        x1 = self.decoder(x1_3, x1_2, x1_1, contour1)  # bs 32 64 64
        x1 = self.cbam0(x1)  # bs 32 64 64

        x2 = self.decoder(x2_3, x2_2, x2_1, contour2)
        x2 = self.cbam1(x2)  # bs 32 64 64
        graph_diff = torch.abs(x1 - x2)
        graph_diff = self.upsample4x(graph_diff)

        '''Transformer分支'''
        token1 = self.get_token(tr_in)
        token2 = self.get_token(tr_in_img2)
        token1_2 = self.cross_transformer_encoder(token1, token2)
        token2_1 = self.cross_transformer_encoder(token2, token1)
        x3 = self.cross_transformer_decoder(tr_in, token1_2).contiguous()  # 32 64 64
        x4 = self.cross_transformer_decoder(tr_in_img2, token2_1).contiguous()  # 32 64 64
        trans_diff = torch.abs(x3 - x4)
        trans_diff = self.upsample4x(trans_diff)  # 32 64 64

        pred = self.classifier(torch.cat([graph_diff, trans_diff], dim=1))
        return pred


if __name__ == '__main__':
    model = CTT_CGINet(out_ch=2, pretrain=False, ckpt_path=None).cuda()
    img1 = torch.randn(1, 3, 256, 256).cuda()
    img2 = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(img1, img2))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
