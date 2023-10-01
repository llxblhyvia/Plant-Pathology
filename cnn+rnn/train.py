from model import CNNRNN
import torch
from data_preprocess import dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

hidden_size = 512
num_epochs = 50
img_size = 224
n_class = 6


def get_acc(out, label_y):
    total = out.shape[0]
    _, pred_label = out.max(1)
    num_correct = (pred_label == label_y).sum().data[0]
    return num_correct / total


# 加载数据
train_dataloader, valid_dataloader = dataloader(batch_size=30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the decoder.
cnnrnn = CNNRNN(img_size, hidden_size, n_class)

# Move the decoder to GPU if CUDA is available.
cnnrnn = cnnrnn.to(device)

# Move last batch of captions (from Step 1) to GPU if CUDA is available
# captions = captions.to(device)


# 优化器
optimizer = torch.optim.Adam(cnnrnn.parameters(), lr=0.0005)
# 损失函数
loss_func = torch.nn.MultiLabelSoftMarginLoss()

Loss_list = []
Accuracy_list = []

# 训练和测试
for epoch in range(num_epochs):
    # 批训练

    batch_idx = 0
    for batch_x, batch_y in tqdm(train_dataloader):

        batch_idx = batch_idx + 1

        batch_y = batch_y.type(torch.LongTensor)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # print(torch.cuda.is_available())
        # print(batch_x.type())
        # print(batch_y.type())
        cnnrnn.train()

        output = cnnrnn(batch_x, batch_y)

        # output = torch.from_numpy(output.cpu().data.numpy()).float()
        # batch_y = torch.from_numpy(batch_y.cpu().data.numpy()).float()

        # print("output",output.size())
        # print("batch_y",batch_y.size())
        # print("n_class",n_class)
        # print("output",output)
        # print("batch_y",batch_y)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        if batch_idx % 433 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch,  loss.item()))
            # for para in cnnrnn.parameters():
            #     print(para)
            _, pred = torch.max(output.data, 1)

            pred = pred.cpu().numpy()
            l_y = torch.max(batch_y, 1)[1].cpu().data.numpy()
            print("train accuracy:", (pred == l_y).sum()/len(l_y))

            Loss_list.append(loss.item())
            Accuracy_list.append(100 * (pred == l_y).sum()/len(l_y))

y_acc = 0
y_len = 0
# 预测
for i, data in enumerate(valid_dataloader):
    # forward
    inputs, labels = data
    labels = labels.type(torch.LongTensor)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = cnnrnn(inputs, labels)

    _, predicted = torch.max(outputs.data, 1)

    predicted = predicted.cpu().numpy()
    label_y = torch.max(labels, 1)[1].cpu().data.numpy()
    y_acc += (predicted == label_y).sum()
    y_len += len(label_y)

    # print("predicted", y_acc)
    # print("len", y_len)
    # print("pred", predicted)
    # print(label_y)

print("test accuracy", y_acc/y_len)

x1 = range(0, len(Accuracy_list))
x2 = range(0, len(Loss_list))
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.show()
plt.savefig("accuracy_loss.jpg")

