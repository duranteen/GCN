from torch import nn
import torch
from torch.nn import init


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """

        :param input_dim: 节点输入维度
        :param output_dim:  输出维度
        :param use_bias: 是否使用偏置项
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_features):
        """

        :param adjacency:  L
        :param input_features: 节点特征
        :return: 图卷积层输出
        """
        output = torch.mm(input_features, self.weight)
        output = torch.sparse.mm(adjacency, output)
        if self.use_bias:
            output += self.bias
        return output