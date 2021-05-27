import contextlib
import glob
import math
import os

import numpy as np
import scipy.signal
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable
from xgboost import XGBClassifier

import model_xgb
import utils
from controller import Controller
from model_xgb import construct_xgb
from read_data import read_data

def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim
def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]
class Trainer(object):
    def __init__(self,args):
        self.args=args
        self.controller_step=0
        self.epoch=0
        self.shared_step=0
        self.start_epoch=0
        self.build_model()
        controller_optimizer=_get_optimizer(self.args.controller_optim)
        self.controller_optim=controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr
        )
        self.ce=nn.CrossEntropyLoss()
        self.X_train=read_data()[0]
        self.X_test = read_data()[1]
        self.y_train = read_data()[2]
        self.y_test = read_data()[3]
        self.accuracy_list=[]
    def build_model(self):
        self.controller=Controller(self.args)
        params, log_probs, entropies = self.controller.sample()
        self.shared = model_xgb.construct_xgb(params)

    def train_shared(self):
        params, log_probs, entropies = self.controller.sample()
        self.shared = model_xgb.construct_xgb(params)
        self.shared.fit(self.X_train.astype('float') / 256, self.y_train)

    def train(self):
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            # self.train_shared()

            # 2. Training the controller parameters theta

            self.train_controller()
        plt.plot(range(self.args.max_epoch*self.args.controller_max_step), self.accuracy_list)
        plt.show()


    def get_loss(self,params):
        self.shared = model_xgb.construct_xgb(params)
        self.shared.fit(self.X_train.astype('float') / 256, self.y_train)
        loss=0
        #用param初始化xgboost并计算loss
        pred=self.shared.predict(self.X_test)
        sample_loss=-accuracy_score(self.y_test,pred)
        loss+=sample_loss
        self.accuracy_list.append(loss)
        return loss

    def get_reward(self,params,entropies):
        entropies=entropies.detach().numpy()
        valid_loss=self.get_loss(params)
        valid_ppl=math.exp(valid_loss)
        R=self.args.reward_c/valid_ppl
        rewards=R+self.args.entropy_coeff*entropies
        return rewards

    def train_controller(self):
        model=self.controller
        model.train()
        avg_reward_base=None
        baseline=None
        reward_history=[]
        entropy_history=[]
        adv_history=[]
        total_loss=0
        valid_idx=0
        for step in range(self.args.controller_max_step):
            params, log_probs, entropies=self.controller.sample()
            rewards=self.get_reward(params,entropies)
            if 1>self.args.discount>0:
                rewards=discount(rewards,self.args.discount)
            reward_history.extend(rewards)
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards
            adv = rewards - baseline
            adv_history.extend(adv)
            loss = -log_probs * utils.get_variable(adv,requires_grad=False)
            loss=loss.sum()
            self.controller_optim.zero_grad()
            loss.backward()
            self.controller_optim.step()
            self.controller_step += 1

