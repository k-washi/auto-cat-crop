
from omegaconf import base
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import optuna
from torch.nn.modules.activation import RReLU
optuna.logging.disable_default_handler()


class Net(nn.Module):
    def __init__(self, cnf):
        super(Net, self).__init__()

        kernel_size = cnf.model.kernel_size
        input_size = cnf.model.input_size

        conv_num = int(cnf.model.conv_num)
        base_channel_num = 3
        next_channel_num = cnf.model.num_filters[0]
        # 畳み込みレイヤに関して
        conv_layer = []
        stride = 1
        padding = 1
        dillation = 1

        filter_size = int(input_size)
        for i in range(0, conv_num - 1):
            conv_layer.append(nn.Conv2d(base_channel_num, next_channel_num,
                                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dillation))

            conv_layer.append(nn.RReLU())
            conv_layer.append(nn.MaxPool2d(2))  # 切り捨て

            base_channel_num, next_channel_num = next_channel_num, cnf.model.num_filters[
                i + 1]

            # conv, maxpool2dの計算
            filter_size = self._output_shape(
                filter_size, kernel_size, stride, dillation, padding)
            filter_size = math.floor(filter_size / 2.)

        # 最後のMaxPool2dは必要ない
        conv_layer.append(nn.Conv2d(base_channel_num, next_channel_num,
                                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=dillation))

        conv_layer.append(nn.RReLU())
        base_channel_num, next_channel_num = next_channel_num, -1
        filter_size = self._output_shape(
            filter_size, kernel_size, stride, dillation, padding)
        # filter_size = math.floor(filter_size / 2.) # maxpoolなし

        self.conv_layer = nn.ModuleList(conv_layer)

        # FCに渡す際のベクトル化に関して
        avg_padding = 0
        avg_kernel_size = 2
        avg_stride = 2
        self.avg = nn.AvgPool2d(kernel_size=avg_kernel_size, stride=avg_stride,
                                padding=avg_padding)  # ceil_mode: False

        # avg pooling のフィルターサイズの計算
        filter_size = math.floor(
            (filter_size + 2 * avg_padding - avg_kernel_size) / avg_stride + 1)
        # conv_output_channel_num = out_size ** 2 * cnf.model.num_filters[-1]
        base_channel_num = base_channel_num * filter_size ** 2

        # FC layer
        fc_layer = [
            nn.Linear(base_channel_num, cnf.model.mid_units[0]),
            nn.RReLU(),
            nn.Linear(cnf.model.mid_units[0], cnf.model.mid_units[1]),
            nn.RReLU()
        ]
        self.fc_layer = nn.ModuleList(fc_layer)
        self.last_fc = nn.Linear(cnf.model.mid_units[1], cnf.model.label_num)

    def _output_shape(self, input, kernel_size, stride, dilation, padding):
        return (input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

    def forward(self, x):
        for i in range(len(self.conv_layer)):
            x = self.conv_layer[i](x)
        x = self.avg(x)
        x = x.view(x.size()[0], -1)

        for i in range(len(self.fc_layer)):
            x = self.fc_layer[i](x)

        x = self.last_fc(x)

        return x
