from model import CNNRNN
import torch
from data_preprocess import dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import time
import numpy as np

hidden_size = 512
num_epochs = 100
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
# 准确率函数
def acc_func(logits, labels):
    _, preds = torch.max(logits.data, 1)
    preds = preds.cpu().numpy()
    labels = torch.max(labels, 1)[1].cpu().data.numpy()
    #preds = torcg.argmax(logits).flatten().cpu().numpy()
    return accuracy_score(labels, preds)

def train(dataloader):
    epoch_loss = []
    logits = []
    labels = []
    cnnrnn.train()
    for batch_x, batch_y in tqdm(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        output = cnnrnn(batch_x, batch_y)
        loss = loss_func(output, batch_y)
        epoch_loss.append(loss.item())
        logits.append(output)
        labels.append(batch_y)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    epoch_acc = acc_func(logits, labels)
    return np.mean(epoch_loss), epoch_acc

    
def evaluate(dataloader):
    epoch_loss = []
    logits = []
    labels = []

    cnnrnn.eval()
    for batch_x, batch_y in tqdm(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        output = cnnrnn(batch_x, batch_y)
        loss = loss_func(output, batch_y)
        epoch_loss.append(loss.item())
        logits.append(output)
        labels.append(batch_y)

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    epoch_acc = acc_func(logits, labels)
    return epoch_loss.mean(), epoch_acc


# 训练和测试
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc = train(train_dataloader)
    valid_loss, valid_acc = evaluate(valid_dataloader)
    end = time.time()

    if valid_loss< best_valid_loss:
        torch.save(cnnrnn.state_dict(), '/nfsshare/home/dl03/cnn+rnn/cnnrnn_model.bin')
        best_valid_loss = valid_loss
    
    run_time = end-start
    print(f'Epoch: {epoch+1:02} | Epoch Time: {run_time//60}m {run_time%60}s')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%')
    print(f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc*100:.2f}%')

