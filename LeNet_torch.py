import torch
import torchvision  # 数据库模块
import torch.nn as nn
import torch.utils.data as Data
import numpy as np


# 超参数
EPOCH = 1 # 所有数据训练次数
BATCH_SIZE = 30
LR = 0.001

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist_data/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download = True
)# 没下载就下载, 下载了就不用再下了

test_data = torchvision.datasets.MNIST(root='./mnist_data/',train=False)
test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)/255
# shape from (10000,28,28)to (10000,1,28,28),value in range(0,1)
test_y = test_data.test_labels
#批训练30samples，1 channel，28×28（30，1，28，28）
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)  # [28,28,1]--->[24,24,6]--->[12,12,6]
        conv2_output = self.conv2(conv1_output)  # [12,12,6]--->[8,8,,16]--->[4,4,16]
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)  # 将[n,4,4,16]维度转化为[n,4*4*16]
        fc1_output = self.fc1(conv2_output)  # [n,256]--->[n,120]
        fc2_output = self.fc2(fc1_output)  # [n,120]-->[n,84]
        fc3_output = self.fc3(fc2_output)  # [n,84]-->[n,10]
        return fc3_output

model=LeNet()
# print(model) # 打印网络构架

optimizer = torch.optim.Adam(model.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()


# training and testing
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        output = model(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('Epoch: ', epoch,"|第%d次训练，loss为%.3f" % (step, loss))


test_output = model(test_x[:10])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')