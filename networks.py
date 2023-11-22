from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class LSTM(nn.Module):
    """LSTM class"""

    def __init__(self, input_size, hidden_size, output_size, batch_size=1, num_layers=1, device="cpu"):
        """
        :param input_size: number of input coming in
        :param hidden_size: number of he hidden units
        :param output_size: size of the output
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        # LSTM

        self.arch = nn.LSTM(self.input_size, self.hidden_size, self.num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.hidden = self.init_hidden()

    def forward(self, x):
        out, self.hidden = self.arch(x, self.hidden)

        output = self.fc(out[-1])  # many to one
        output = nnf.log_softmax(output, dim=1)
        return output

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))

    def get_wl0(self):
        wi, wf, wg, wo = self.arch.weight_ih_l0.reshape(4, self.hidden_size, -1)
        wl0 = {"WI": wi,
               "WF": wf,
               "WG": wg,
               "WO": wo}
        return wl0

    def get_wl1(self):
        wi, wf, wg, wo = self.arch.weight_ih_l1.reshape(4, self.hidden_size, -1)
        wl1 = {"WI": wi,
               "WF": wf,
               "WG": wg,
               "WO": wo}
        return wl1

    def get_bl0(self):
        bi, bf, bg, bo = self.arch.bias_ih_l0.reshape(4, self.hidden_size, -1)
        bl0 = {"BI": bi,
               "BF": bf,
               "BG": bg,
               "BO": bo}
        return bl0

    def get_bl1(self):
        bi, bf, bg, bo = self.arch.bias_ih_l1.reshape(4, self.hidden_size, -1)
        bl1 = {"BI": bi,
               "BF": bf,
               "BG": bg,
               "BO": bo}
        return bl1


class GRU(nn.Module):
    """GRU class"""

    def __init__(self, input_size, hidden_size, output_size, batch_size=1, num_layers=1, device="cpu"):
        """
        :param input_size: number of input coming in
        :param hidden_size: number of he hidden units
        :param output_size: size of the output
        """
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        # CORE

        self.arch = nn.GRU(self.input_size, self.hidden_size, self.num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.hidden = self.init_hidden()

    def forward(self, x):
        out, self.hidden = self.arch(x, self.hidden)

        output = self.fc(out[-1])  # many to one
        output = nnf.log_softmax(output, dim=1)
        return output

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)


class RNN(nn.Module):
    """RNN class"""

    def __init__(self, input_size, hidden_size, output_size, batch_size=1, num_layers=1, device="cpu"):
        """
        :param input_size: number of input coming in
        :param hidden_size: number of he hidden units
        :param output_size: size of the output
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        # CORE

        self.arch = nn.RNN(self.input_size, self.hidden_size, self.num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.hidden = self.init_hidden()

    def forward(self, x):
        out, self.hidden = self.arch(x, self.hidden)

        output = self.fc(out[-1])  # many to one
        output = nnf.log_softmax(output, dim=1)
        return output

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

