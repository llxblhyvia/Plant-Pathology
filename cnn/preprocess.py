import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns





def dividelabels(data):
    plt.figure(figsize=(25, 15))
    plt.xlabel("labels", fontsize=15)
    plt.xticks(rotation=20, fontsize=10, fontweight="bold")
    plt.ylabel("count", fontsize=15)
    plt.yticks(fontsize=10)
    sns.barplot(data=data, x=data['labels'].value_counts().index, y=data['labels'].value_counts().values)
    plt.savefig('/nfsshare/home/dl03/1.0/rawfig.jpg', dpi=1000)


    data['labels'] = data['labels'].apply( lambda string: string.split(' ') )


    mlb = MultiLabelBinarizer()
    hot_labels = mlb.fit_transform(data['labels'])
    train_labels = pd.DataFrame(hot_labels, columns=mlb.classes_, index=data.index)
    #print(train_labels.head())
    plt.figure(figsize=(25,10))
    sns.barplot(x=train_labels.columns, y=train_labels.sum().values)
    plt.savefig('/nfsshare/home/dl03/1.0/train_label.jpg', dpi=1000)

if __name__ == '__main__':
    data = pd.read_csv("/nfsshare/home/dl03/1.0/train.csv")
    data.dropna(axis=0)
    dividelabels(data)