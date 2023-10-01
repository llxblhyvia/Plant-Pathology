import torch
import torch.nn as nn
import torchvision.models as models


class CNNRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_class, num_layers=1):
        super(CNNRNN, self).__init__()
        resnet = models.resnet50()
        for param in resnet.parameters():
            param.requires_grad_(True)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)



        # rnn

        self.embedding_layer = nn.Embedding(n_class, embed_size)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, images, batch_y):
        # print(images.size())
        features = self.resnet(images)

        features = features.view(features.size(0), -1)

        features = self.embed(features)

        embed = self.embedding_layer(batch_y)

        embed = torch.cat((features.unsqueeze(1), embed), dim=1)

        lstm_outputs, _ = self.lstm(embed)

        out = self.linear(lstm_outputs[:, 0, :])

        return out

#
# class RNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, n_class, num_layers=1):
#         super().__init__()
#         self.embedding_layer = nn.Embedding(n_class, embed_size)
#
#         self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)
#
#         self.linear = nn.Linear(hidden_size, n_class)
#
#     def forward(self, features, captions):
#         captions = captions[:, :-1]
#         embed = self.embedding_layer(captions)
#         embed = torch.cat((features.unsqueeze(1), embed), dim=1)
#         lstm_outputs, _ = self.lstm(embed)
#         out = self.linear(lstm_outputs)
#
#         return out
#
#
# criterion = nn.CrossEntropyLoss()
#
# optimzier = torch.optim.Adadelta(net.parameters(), 1e-1)
