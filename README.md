# ML_for_learner
该项目旨在使用numpy实现一个类scikit-learn的机器学习库，对于相关的知识，均配有blog文章对其理论进行讲解，对于部分功能，还配有notebook分析代码实现上的细节。该项目的初衷是为那些算法学习者提供从理论到实现的一站式服务。

由于本人学识有限，并且没有Python开发经验，该库目前还是一个非常松散的代码集合体。如果您在blog、notebook或者code中发现任何纰漏或bug，请迅速联系我，当然也可以在项目页面提issue，谢谢。

QQ: 435248055 &ensp; | &ensp; WeChat: QQ435248055 &ensp; | &ensp; [Blog](https://daya-jin.github.io/)

---

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

|Class|Approach|Code|
|-|-|-|
|[Model Selection](https://daya-jin.github.io/2018/12/11/Model_Assessment_and_Selection/)|Dataset Split|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/model_selection/train_test_split.py)|
|[Metrics](https://daya-jin.github.io/2019/03/27/Evaluation_Metircs/)|Accuracy|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/accuracy_score.py)|
||Log loss|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/log_loss.py)|
||F1-score|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/f1_score.py)|
||[AUC](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/AUC.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/roc_auc_score.py)|
||[Euclidean Distances](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/pairwise/euclidean_distances.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/pairwise/euclidean_distances.py)|

## Preprocessing data

|Class|Algorithm|Implementation|Code|
|-|-|-|-|
|[Feature Scaling](https://daya-jin.github.io/2019/03/20/Data_Scaling/)|StandardScaler|[notebook]()|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/preprocessing/StandardScaler.py)|
||MinMaxScaler|||
|Unsupervised dimensionality reduction|PCA|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/decomposition/PCA.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/decomposition/PCA.py)|
|Supervised dimensionality reduction|[Linear Discriminant Analysis](https://daya-jin.github.io/2018/12/05/LinearDiscriminantAnalysis/)|[notebook](https://github.com/Daya-Jin/ML_for_learner/blob/master/discriminant_analysis/LinearDiscriminantAnalysis.ipynb)|[code](https://github.com/Daya-Jin/ML_for_learner/blob/master/discriminant_analysis/LinearDiscriminantAnalysis.py)|

## Known Issues

```/datasets/dataset.py```中所有的```load_*()```函数均存在路径问题，解决该问题待参阅scikit-learn源码。