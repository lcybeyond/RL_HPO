import collections
import os
import torch
import torch.nn.functional as F

import utils


class Controller(torch.nn.Module):
    def __init__(self, args):
        # 初始化父类
        torch.nn.Module.__init__(self)
        self.args = args
        self.encoder = torch.nn.Linear(args.num_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
        self.decoder = torch.nn.Linear(args.controller_hid, 2)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(torch.zeros(key, self.args.controller_hid),requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden):
        embed = self.encoder(inputs)
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoder(hx)
        logits=(self.args.tanh_c*torch.sigmoid(logits))
        return logits,(hx,cx)

    def sample(self):
        inputs=self.static_inputs[1]
        hidden=self.static_init_hidden[1]
        activations=[]
        log_probs=[]
        entropies=[]
        for num in range(self.args.para_num):
            logits,hidden=self.forward(inputs,hidden)
            action = torch.normal(mean=logits[0][0], std=logits[0][1])
            probs = torch.exp(-(action-logits[0][0])**2/logits[0][1]**2)/logits[0][1]
            log_prob = torch.log(probs)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs)
            entropies.append(entropy.unsqueeze(-1))
            log_probs.append(log_prob.unsqueeze(-1))
            activations.append(action)
        activations=torch.stack(activations)
        return activations,torch.cat(log_probs),torch.cat(entropies)
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, requires_grad=False),
                utils.get_variable(zeros.clone(), requires_grad=False))

