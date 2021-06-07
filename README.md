# Fraud Detection
We are trying to improve the accuracy for fraud detection systems which would result in more reliable systems and better consumer experience. The dataset used has been taken from Kaggle. It consists of two separate datasets containing transaction information and identity that can be joined via transaction id. Later, machine learning algorithms are used to obtain a model to predict on the test data.

# Description

The digital age has opened doors to new possibilities. The process of purchasing an item has taken a different route in the past decade or so. However, the rise of new methods brings in new problems. Now-a-days, the likelihood of a person's digital information being stolen is significant. Bank details, credit card information can be obtained by hackers and used for illegal purchases. This calls for the need of fraud detection services to notify consumers of any illegal activity taken place using their identity. The fraud prevention system would result in the saving of a surplus of money of the consumer. Early detection would mean that the proper authorities are notified of the illegal activities as soon as possible. This, in turn, could speed up the process of apprehending those responsible of conducting the illicit activities. After the intial cleaning of the data, the Gaussian Naive Bayes model has been used to attain prediction on the test data.

# Getting Started
### Libraries

```
!pip xgboost
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, QuantileTransformer, RobustScaler, MaxAbsScaler

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
```

### Datasets
The datasets can be obtained from the following link: [Datasets](https://www.kaggle.com/c/ieee-fraud-detection/data)


