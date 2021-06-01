import torch
from torchsummary import summary

import read_data


class estimate_loss_net(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # self.args=args
        self.net1=torch.nn.Linear(10,30)
        self.net2=torch.nn.Linear(30,60)
        self.net3=torch.nn.Linear(60,1)
    def forward(self,input):
        x=self.net1(input)
        x=torch.dropout(x,p=0.2,train=True)
        x=self.net2(x)
        x=torch.dropout(x,p=0.2,train=True)
        x=self.net3(x)
        return x

# abc=estimate_loss_net()
# summary(abc,input_size=[(10,)],batch_size=1)

class estimate_loss():
    def __init__(self):
        self.net=estimate_loss_net()
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.X_train=torch.from_numpy(read_data.read_param_loss()[0]).float()
        self.y_train=torch.from_numpy(read_data.read_param_loss()[2]).float()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=0.02)
    def train(self,steps):
        for i in range(steps):
            re=self.net.forward(self.X_train)
            loss=self.mse(re,self.y_train)
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        torch.save(self.net.state_dict(), './estimate_net')
        print('结束')

    def estimate(self,param):
        return self.net.forward(param)

# abc=estimate_loss()
# abc.train(3000)
