# ML_for_learner
该项目使用numpy实现一个类scikit-learn的机器学习库，对于相关的知识，均配有blog文章对其理论进行讲解，对于部分功能，还配有notebook分析实现上的细节。

该项目的初衷是为那些算法学习者提供一个从理论到实现的全套知识平台。除此之外，该项目不仅限于学习者，我会尽可能的保证该项目中各模块的可用性。由于本人学识有限，如果您在blog、notebook或者code中发现任何纰漏或bug，请迅速联系我，当然也可以在项目页面提issue，谢谢。

QQ: 435248055

WeChat: QQ435248055

Blog: [https://daya-jin.github.io/](https://daya-jin.github.io/)

## Supervised learning

|Class|Algorithm|Implementation|Code|
|-|-|-|-|
|Generalized Linear Models|[Linear Regression](https://daya-jin.github.io/2018/12/01/LinearRegression/)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/linear_model/LinearRegression.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/linear_model/LinearRegression.py)|
||[Logistic regression](https://daya-jin.github.io/2018/12/01/LogisticRegression/)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/linear_model/LogisticRegression.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/linear_model/LogisticRegression.py)|
|[Nearest Neighbors](https://daya-jin.github.io/2018/12/29/KNearestNeighbor/)|Nearest Neighbors Classification|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/neighbors/KNN.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/neighbors/KNeighborsClassifier.py)|
|[Naive Bayes](https://daya-jin.github.io/2018/12/01/NaiveBayes/)|Gaussian Naive Bayes|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/naive_bayes/Gaussian%20Naive%20Bayes.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/naive_bayes/GaussianNB.py)|
|[Support Vector Machine](https://daya-jin.github.io/2018/12/01/SupportVectorMachine/)|SVC|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/svm/SMO.ipynb)||
|[Decision Trees](https://daya-jin.github.io/2018/12/01/DecisionTree/)|[ID3 Classification](https://daya-jin.github.io/2018/12/01/DecisionTree/#id3)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Clf.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Clf.py)|
||ID3 Regression|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Reg.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Reg.py)|
||[CART Classification](https://daya-jin.github.io/2018/12/01/DecisionTree/#classification-and-regression-tree)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeClassifier.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeClassifier.py)|
||[CART Regression](https://daya-jin.github.io/2018/12/01/DecisionTree/#classification-and-regression-tree)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeRegressor.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeRegressor.py)|
|[Ensemble methods](https://daya-jin.github.io/2018/12/01/EnsembleLearning/)|[Random Forests Classification](https://daya-jin.github.io/2018/12/01/EnsembleLearning/#random-forest)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/ensemble/RandomForestClassifier.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/ensemble/RandomForestClassifier.py)|
||Random Forests Regression|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/ensemble/RandomForestRegressor.ipynb)|[code]()|
||[AdaBoosting Classification](https://daya-jin.github.io/2018/12/01/EnsembleLearning/#boosting)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/ensemble/AdaBoostClassifier.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/ensemble/AdaBoostClassifier.py)|

## Unsupervised learning

|Class|Algorithm|Implementation|Code|
|-|-|-|-|
|Gaussian mixture models|[Gaussian Mixture](https://daya-jin.github.io/2019/03/15/Gaussian_Mixture_Models/)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/mixture/GaussianMixture.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/mixture/GaussianMixture.py)|
|Clustering|[K-means](https://daya-jin.github.io/2018/12/01/KMeans/)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/cluster/KMeans.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/cluster/KMeans.py)|

## Model selection and evaluation

|Class|Approach|code|
|-|-|-|
|metrics|||
||||
||||
||||
||||
||||
||||

## Preprocessing data

|Class|Algorithm|Implementation|Code|
|-|-|-|-|
|[Scaling features](https://daya-jin.github.io/2019/03/20/Data_Scaling/)|StandardScaler|[notebook]()|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/preprocessing/StandardScaler.py)|
||MinMaxScaler|||
|Unsupervised dimensionality reduction|PCA|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/decomposition/PCA.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/decomposition/PCA.py)|