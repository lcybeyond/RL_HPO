import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import torch
import matplotlib.pyplot as plt
import read_data
from sklearn import preprocessing
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Net, self).__init__()
        self.embedding1 = nn.Linear(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=4)
        self.embedding2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedding = F.relu(F.normalize(self.embedding1(x).unsqueeze(1)))

        output, (hidden, cell) = self.rnn(embedding)

        output=self.embedding2(output).squeeze(-1)

        return output

rnn = Net(2, 10, 10)
optimizer = optim.Adam(rnn.parameters(), lr=0.0001)

accuracy_list=[]

def reward(pred):
    bound_list = [[1, 25],
                  [0.001, 0.1],
                  [50, 1200],
                  [0.05, 0.9],
                  [1, 9],
                  [0.5, 1],
                  [0.5, 1],
                  [0.5, 1],
                  [0.1, 0.9],
                  [0.01, 0.1]]
    # dataset = loadtxt('./HTRU2/HTRU_2.csv', delimiter=",")
    #
    # X = dataset[:, 0:8]
    # Y = dataset[:, 8]
    #
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    para_ = []
    para_init=np.empty(10)
    para_list = pred.clone().detach().numpy()
    for i in range(10):
        mean = para_list[i][0][0]
        mean = (1 - math.e ** (-2 * mean)) / (1 + math.e ** (-2 * mean))
        if para_list[i][0][1]<0:
            para_list[i][0][1]=1
        para = np.random.normal(mean, para_list[i][0][1], 1)
        para_init[i]=para
        para = float(bound_list[i][0] + (bound_list[i][1] - bound_list[i][0]) * (1 + para) / 2)
        if i == 0 or i == 2 or i == 4:
            para = int(para)
        if para > bound_list[i][1]:
            para = bound_list[i][1]
        elif para < bound_list[i][0]:
            para = bound_list[i][0]
        para_.append(para)
    #print('para_', para_)
    model = XGBClassifier(max_depth=para_[0],
                          learning_rate=para_[1],
                          n_estimators=para_[2],
                          gamma=para_[3],
                          min_child_weight=para_[4],
                          subsample=para_[5],
                          colsample_bytree=para_[6],
                          colsample_bylevel=para_[7],
                          reg_alpha=para_[8],
                          reg_lambda=para_[9])
    # model.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = read_data.read_data()
    model.fit(X_train.astype('float') / 256, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)
    print('accuracy',accuracy)
    R = accuracy * 100.0 - 70
    loss = torch.Tensor([0])

    #print(loss)

    for i in range(10):
        mean = pred[i][0][0]
        # mean = (1 - math.e ** (-2 * mean)) / (1 + math.e ** (-2 * mean))
        variance = pred[i][0][1]
        print("mean  variance para_[i]",mean,variance,para_init[i])
        loss = torch.add(loss, -((para_init[i] - mean) ** 2)/(variance**2) -torch.log( variance))
        #print(torch.log(((para_[i] - mean) ** 2) / variance))
    loss = -loss * R
    print("loss",loss)
    return loss

def train(rnn, iterator, optimizer):
    x=torch.ones((10,2))
    #print(x)
    for i in range(iterator):
        pred = rnn(x)
        print('pred',pred)
        #print(pred)
        loss=reward(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(pred)
train(rnn,10,optimizer)
print(accuracy_list)
plt.plot(range(10),accuracy_list)
plt.show()


