
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# network definition
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

iris = datasets.load_iris()
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
y[np.arange(len(iris.target)), iris.target] = 1
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.25)
x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)
y = Variable(torch.from_numpy(y_train).float())
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(3000):
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(i)

# test
outputs = net(Variable(torch.from_numpy(X_test).float()))
_, predicted = torch.max(outputs.data, 1)
y_predicted = predicted.numpy()
y_true = np.argmax(y_test, axis=1)
accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
print('accuracy: {0}%'.format(accuracy))


# utility function to predict for an unknown data
def predict(X):
    X = Variable(torch.from_numpy(np.array(X)).float())
    outputs = net(X)
    return np.argmax(outputs.data.numpy())
