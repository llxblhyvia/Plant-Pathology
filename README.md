# Plant Pathology
> Course project of "Deep Learning", 2021 Spring.

## Abstract
The task is from the Kaggle competition ["plant-pathology"](https://www.kaggle.com/c/plant-pathology-2021-fgvc8/). It aims to identify multiple diseases on plant leaves.

According to our [EDA](EDA.ipynb) and the research in botany, we found that it seems that there are no correlation between diverse diseases on leaves but we still want to build our model to examine the theory. So we choose two kinds of assumptions to build our CV model.
One assuming there are correlation between different diseases while one not. The former is based solely on CNN--different versions of ResNet and DenseNet 169. The latter is based on CNN-RNN architecture.

- Built several [CNN](cnn) models(ResNet50, ResNet101,DenseNet169) based on the assumption that labels were independent; achieved the highest accuracy of 0.9046 with DenseNet169.
- Proposed [another assumption](cnn+rnn) that labels were correlated and applied LSTM and CNN to process texts and images respectively; utilized multi-label soft margin loss as the loss function and achieved an accuracy of 0.8013.

## File Introduction
[EDA](EDA.ipynb) is the Exploratory data analysis of training data including distribution visualization and corralation analysis.

[CNN](cnn) is the assuming-independent model including [data processing](cnn/preprocess.py), [train data](cnn/train.csv), and the three models.

[CNN+RNN](cnn+rnn) is the assuming-dependent  model including standard CV python files: [data processing](cnn+rnn/data_preprocess.py), [train](cnn+rnn/train.py), [test](cnn+rnn/test.py), [model](cnn+rnn/model.py) etc.

## Conclusion
According to the model performance that the no-correlation modeling outperforms correlation modeling a lot, proving the theory and our EDA results.
