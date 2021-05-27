import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from agent import output_
import math
import torch

dataset = loadtxt('./HTRU2/HTRU_2.csv', delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#max depth ⇒ n estimators ⇒ min child weight ⇒ colsample bytree ⇒ reg alpha ⇒ learning rate ⇒ gamma ⇒ subsample ⇒ colsample bylevel ⇒ reg lambda

bound_list=[[1,25],
            [0.001,0.1],
            [50,1200],
            [0.05,0.9],
            [1,9],
            [0.5,1],
            [0.5,1],
            [0.5,1],
            [0.1,0.9],
            [0.01,0.1]]
#bound_list=torch.Tensor(bound_list)

output=output_()
para_list = output.detach().numpy()
print(para_list[0])
mean=para_list[0][0][0]
mean=(1-math.e**(-2*mean))/(1+math.e**(-2*mean))
MAXDEPTH=np.random.normal(mean,para_list[0][0][1],1)
MAXDEPTH=int(1+24*(1+MAXDEPTH)/2)
print('MAXDEPTH',MAXDEPTH)

para_=[]
for i in range(10):
    mean = para_list[i][0][0]
    mean = (1 - math.e ** (-2 * mean)) / (1 + math.e ** (-2 * mean))
    para = np.random.normal(mean, para_list[i][0][1], 1)
    para = float(bound_list[i][0] +  (bound_list[i][1]-bound_list[i][0])* (1 + para) / 2)
    if i==0 or i==2 or i==4:
        para=int(para)
    if para>bound_list[i][1]:
        para=bound_list[i][1]
    elif para<bound_list[i][0]:
        para=bound_list[i][0]
    para_.append(para)
    print('para_', para_)

MAXDEPTH=3
NESTIMATORS=100
MINCHILDWEIGHT=1
COLSAMPLEBYTREE=0.8
REGALPHA=0
LEARNINGRATE=0.1
GAMMA=0
SUBSAMPLE=1
COLSAMPLEBYLEVEL=1
REGLAMBDA=1

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
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

reward=accuracy*100.0-95
loss=torch.Tensor([0])

print(loss)

for i in range(10):
    mean = output[i][0][0]
    mean = (1 - math.e ** (-2 * mean)) / (1 + math.e ** (-2 * mean))
    variance=output[i][0][1]**2
    loss = torch.add(loss,torch.log(((para_[i]-mean)**2)/variance))
    print(torch.log(((para_[i]-mean)**2)/variance))
    print(loss)
loss=-loss*reward
print(loss)
loss.backward()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


