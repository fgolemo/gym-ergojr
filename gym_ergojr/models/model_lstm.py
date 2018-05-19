import torch.nn.functional as F
from torch import nn, autograd, torch


class ModelLstm(nn.Module):
    def __init__(self, n_input_state_sim=12, n_input_state_real=12, n_input_actions=6, nodes=256, layers=5):
        super().__init__()

        self.n_nodes = nodes
        self.n_layers = layers

        self.linear1 = nn.Linear(n_input_state_sim + n_input_state_real + n_input_actions, nodes)
        self.lstm1 = nn.LSTM(nodes, nodes, layers)
        self.linear2 = nn.Linear(nodes, n_input_state_real)

        self.hidden = self.init_hidden()

    def zero_hidden(self):
        # print(dir(self.hidden[0]))
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = autograd.Variable(torch.zeros(self.n_layers, 1, self.n_nodes), requires_grad=False)
        c = autograd.Variable(torch.zeros(self.n_layers, 1, self.n_nodes), requires_grad=False)

        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, data_in):
        # the view is to add the minibatch dimension (which is 1)
        # print(data_in.view(1, 1, -1).size())
        # import ipdb; ipdb.set_trace()

        out = torch.cat((data_in[0], data_in[1], data_in[2]), dim=2)
        out = F.leaky_relu(self.linear1(out))
        out, self.hidden = self.lstm1(out.permute((1, 0, 2)), self.hidden)
        out = F.leaky_relu(out.permute((1, 0, 2)))
        out = self.linear2(out)
        return out
