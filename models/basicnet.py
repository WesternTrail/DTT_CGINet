import torch
from torch import nn, einsum
import torch.nn.functional as F
import os
import h5py, math
import numpy as np
from einops import rearrange


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1. / math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x out_c

        adj = torch.softmax(adj, dim=2) # (kxk)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim) # （32，32）
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert (loop == 1 or loop == 2 or loop == 3) # loop == 2
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x))  # b x c x k,adj:kxk
        x = self.relu(x) # (b,c,k)
        return x


class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num # 投影的图中的顶点个数
        self.dim = dim # 图中每个顶点的特征维度C,应该与特征图的C相同
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim)) # 参数化W矩阵
        self.sigma = nn.Parameter(torch.rand(node_num, dim))  # 参数化西格玛矩阵

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H * W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)  # 1, 32, 4096
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(
                sigma[node_id, :])  # + eps) #（B,hw,C）
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1) # 经过一个softmax进行处理 (num_vertex,hw)，计算特征图上每个位置分配给当前顶点的概率

        return soft_assign

    def forward(self, x):
        # B 32 64 64
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        sigma = torch.sigmoid(self.sigma)  # (num_vertex,num_dim)
        soft_assign = self.gen_soft_assign(x, sigma)  # 获得软分配q，计算x到anchor的软分配。1 32 3600, # B x num_vertex x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)  # 1 32 32
        for node_id in range(self.node_num): #
            aa = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # 1 4096 32
            bb = self.anchor[node_id, :]  # 32
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(
                sigma[node_id, :])  # + eps) B 4096 32
            c = soft_assign[:, node_id, :]  # B 4096
            cc = soft_assign[:, node_id, :].unsqueeze(2)  # B 4096 1
            d = residual.mul(soft_assign[:, node_id, :].unsqueeze(2))  # （B,4096,32） X (B,4096,1) = (B,4096,32)
            dd = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1)  # 1 512
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (
                    soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps) # 计算图的顶点特征矩阵。防止分母为0 （num_vertex,C）

        nodes = F.normalize(nodes, p=2, dim=2)  # intra-normalization，计算zk2的l2范数
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1)  # l2 normalize，除以l2范数？

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign

