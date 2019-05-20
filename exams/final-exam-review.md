# Final Exam Review

## Supervised vs Unsupervised Learning
- **Supervised:** All data is labeled and the algorithms learn to predict the output from the input data.
- **Unsupervised:** All data is unlabeled and the algorithms learn to inherent structure from the input data.
- **Semi-supervised:** Some data is labeled but most of it is unlabeled and a mixture of supervised and unsupervised techniques can be used.

[[1](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/)]

---

## Sampling (With Replacement, Without Replacement, Stratified).
- In **sample with replacement**, the two sample values are independent. Practically, this means that what we get on the first one doesn't affect what we get on the second. Mathematically, this means that the covariance between the two is zero. When sampling is performed with replacement, it’s called bagging

- In **sampling without replacement**, the two sample values aren't independent. Practically, this means that what we got on the for the first one affects what we can get for the second one. Mathematically, this means that the covariance between the two isn't zero. When sampling is performed without replacement, it’s called pasting.

- In **stratified sampling**, the researcher divides the population into separate groups, called strata. Then, a probability sample (often a simple random sample ) is drawn from each group to guarantee the test set is representative of the overall population. Stratified sampling has several advantages over simple random sampling.

[[1](https://web.ma.utexas.edu/users/parker/sampling/repl.htm)] [[2](https://stattrek.com/statistics/dictionary.aspx?definition=stratified_sampling)]

---

## Model Overfitting/Underfitting
### Overfitting
Overfitting refers to a model that performs well on training data but does not generalize well.

Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize.

Overfitting is more likely with nonparametric and nonlinear models that have more flexibility when learning a target function. As such, many nonparametric machine learning algorithms also include parameters or techniques to limit and constrain how much detail the model learns.

For example, decision trees are a nonparametric machine learning algorithm that is very flexible and is subject to overfitting training data. This problem can be addressed by pruning a tree after it has learned in order to remove some of the detail it has picked up.

Possible solutions are:
- Simplify the model by selecting one with fewer parameters
- Gather more training data
- Reduce noise in the training data (fix data errors, remove outliers etc.)

### Underfitting
Underfitting refers to a model that can neither model the training data nor generalize to new data (too simple to learn the underlying structure of the data)

An underfit machine learning model is not a suitable model and will be obvious as it will have poor performance on the training data.

Underfitting is often not discussed as it is easy to detect given a good performance metric. The remedy is to move on and try alternate machine learning algorithms. Nevertheless, it does provide a good contrast to the problem of overfitting.

Possible solutions are:
- Select a more powerful model with more parameters
- Feed better features to the learning algorithm
- Reducing the constraints of the model

[[1](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)]

----

## One Hot Encoding
One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.
One hot encoding creates new (binary) columns, indicating the presence of each possible value from the original data. After one-hot encoding, we get a matrix of thousands of columns, and the matrix is full of zeros, except for a single 1 per row.

[[1](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)]

---

## Classification
- **Decision tree** builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy). Leaf node (e.g., Play) represents a classification or decision. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data. 		
- **Gradient descent** is based on the observation that if a multi-variate function F is defined and differentiable in a neighbourhood of a point x, then the function F decreases fastest if one goes from the point x in the direction of the negative gradient of F at the point x.
- **Random forest** builds multiple decision trees and merges them together to get a more accurate and stable prediction.

- In machine learning, **multiclass or multinomial classification** is the problem of classifying instances into one of three or more classes. (Classifying instances into one of two classes is called binary classification.) While some classification algorithms naturally permit the use of more than two classes, others are by nature binary algorithms; these can, however, be turned into multinomial classifiers by a variety of strategies.
    - **One-vs.-rest: (or one-vs.-all, OvA or OvR, one-against-all, OAA) strategy** involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. This strategy requires the base classifiers to produce a real-valued confidence score for its decision, rather than just a class label; discrete class labels alone can lead to ambiguities, where multiple classes are predicted for a single sample.
    - In the **one-vs.-one (OvO) reduction**, one trains K (K − 1) / 2 binary classifiers for a K-way multiclass problem; each receives the samples of a pair of classes from the original training set, and must learn to distinguish these two classes. At prediction time, a voting scheme is applied: all K (K − 1) / 2 classifiers are applied to an unseen sample and the class that got the highest number of "+1" predictions gets predicted by the combined classifier. Like OvR, OvO suffers from ambiguities in that some regions of its input space may receive the same number of votes.
- In machine learning, **multilabel classification** and the strongly related problem of multi-output classification are variants of the classification problem where multiple labels may be assigned to each instance. Multi-label classification is a generalization of multiclass classification, which is the single-label problem of categorizing instances into precisely one of more than two classes; in the multi-label problem there is no constraint on how many of the classes the instance can be assigned to. Formally, multi-label classification is the problem of finding a model that maps inputs x to binary vectors y (assigning a value of 0 or 1 for each element (label) in y).

[[Decision Tree](https://www.saedsayad.com/decision_tree.htm)] [[Gradient Descent](https://towardsdatascience.com/gradient-descent-demystified-bc30b26e432a)] [[Random Forest](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd)] [[Multiclass](https://en.wikipedia.org/wiki/Multiclass_classification)] [[Multilabel](https://en.wikipedia.org/wiki/Multi-label_classification)]

---

## Linear vs Nonlinear Regression
- **Linear Regression** is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of independent variables being used.
- **Nonlinear regression** is a statistical technique that helps describe nonlinear relationships in experimental data. Nonlinear regression models are generally assumed to be parametric, where the model is described as a nonlinear equation. Typically machine learning methods are used for non-parametric nonlinear regression.

[[Linear Regression](https://www.geeksforgeeks.org/ml-linear-regression/)] [[Nonlinear Regression](https://www.mathworks.com/discovery/nonlinear-regression.html)]

## Model Evaluation
### Confusion Matrix
- In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one (in unsupervised learning it is usually called a matching matrix). Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another). 

It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).

### MSE
In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and what is estimated. MSE is a risk function, corresponding to the expected value of the squared error loss. The fact that MSE is almost always strictly positive (and not zero) is because of randomness or because the estimator does not account for information that could produce a more accurate estimate.

The MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.

The MSE is the second moment (about the origin) of the error, and thus incorporates both the variance of the estimator (how widely spread the estimates are from one data sample to another) and its bias (how far off the average estimated value is from the truth). For an unbiased estimator, the MSE is the variance of the estimator. Like the variance, MSE has the same units of measurement as the square of the quantity being estimated. In an analogy to standard deviation, taking the square root of MSE yields the root-mean-square error or root-mean-square deviation (RMSE or RMSD), which has the same units as the quantity being estimated; for an unbiased estimator, the RMSE is the square root of the variance, known as the standard error.

### Error Rate
One of the advantages of supervised learning is that we can use testing sets to get an objective measurement of learning performance.

The inaccuracy of predicted output values is termed the **error** of the method.

If target values are categorical, the error is expressed as an **error rate**.

This is the proportion of cases where the prediction is wrong.

### Precision & Recall
In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Both precision and recall are therefore based on an understanding and measure of relevance.

Suppose a computer program for recognizing dogs in photographs identifies 8 dogs in a picture containing 12 dogs and some cats. Of the 8 identified as dogs, 5 actually are dogs (true positives), while the rest are cats (false positives). The program's precision is 5/8 while its recall is 5/12. When a search engine returns 30 pages only 20 of which were relevant while failing to return 40 additional relevant pages, its precision is 20/30 = 2/3 while its recall is 20/60 = 1/3. So, in this case, precision is "how useful the search results are", and recall is "how complete the results are".

In statistics, if the null hypothesis is that all items are irrelevant (where the hypothesis is accepted or rejected based on the number selected compared with the sample size), absence of type I and type II errors (i.e.: perfect sensitivity and specificity of 100% each) corresponds respectively to perfect precision (no false positive) and perfect recall (no false negative). The above pattern recognition example contained 8 − 5 = 3 type I errors and 12 − 5 = 7 type II errors. Precision can be seen as a measure of exactness or quality, whereas recall is a measure of completeness or quantity. The exact relationship between sensitivity and specificity to precision depends on the percent of positive cases in the population.

In simple terms, high precision means that an algorithm returned substantially more relevant results than irrelevant ones, while high recall means that an algorithm returned most of the relevant results.

[[Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)] [[MSE](https://en.wikipedia.org/wiki/Mean_squared_error)] [[Error Rate](http://users.sussex.ac.uk/~christ/crs/ml/lec03a.html)] [[Precision & Recall](https://en.wikipedia.org/wiki/Precision_and_recall)]

---

## Deep Learning & DNN
### Deep Learning
Deep learning is a class of machine learning algorithms that: use multiple layers to progressively extract higher level features from raw input. For example, in image processing, lower layers may identify edges, while higher layer may identify human-meaningful items such as digits/letters or faces.


### Deep Neutral Networks (DNN)
A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers.[11][2] The DNN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output. For example, a DNN that is trained to recognize dog breeds will go over the given image and calculate the probability that the dog in the image is a certain breed. The user can review the results and select which probabilities the network should display (above a certain threshold, etc.) and return the proposed label. Each mathematical manipulation as such is considered a layer, and complex DNN have many layers, hence the name "deep" networks.

DNNs can model complex non-linear relationships. DNN architectures generate compositional models where the object is expressed as a layered composition of primitives. The extra layers enable composition of features from lower layers, potentially modeling complex data with fewer units than a similarly performing shallow network.

Deep architectures include many variants of a few basic approaches. Each architecture has found success in specific domains. It is not always possible to compare the performance of multiple architectures, unless they have been evaluated on the same data sets.

DNNs are typically feedforward networks in which data flows from the input layer to the output layer without looping back. At first, the DNN creates a map of virtual neurons and assigns random numerical values, or "weights", to connections between them. The weights and inputs are multiplied and return an output between 0 and 1. If the network didn’t accurately recognize a particular pattern, an algorithm would adjust the weights. That way the algorithm can make certain parameters more influential, until it determines the correct mathematical manipulation to fully process the data.

[[1](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks)]

---

## CNN vs RNN 
- A **CNN (Convolutional Neural Networks)** will learn to recognize patterns across space. So, as you say, a CNN will learn to recognize components of an image (e.g., lines, curves, etc.) and then learn to combine these components to recognize larger structures (e.g., faces, objects, etc.
    - CNN take a fixed size input and generate fixed-size outputs.
    - CNN is a type of feed-forward artificial neural network - are variations of multilayer perceptrons which are designed to use minimal amounts of preprocessing.
    - CNNs use connectivity pattern between its neurons is inspired by the organization of the animal visual cortex, whose individual neurons are arranged in such a way that they respond to overlapping regions tiling the visual field.
    - CNNs are ideal for images and videos processing.
- A **RNN (Recurrent Neural Networks)** will similarly learn to recognize patterns across time. So a RNN that is trained to translate text might learn that "dog" should be translated differently if preceded by the word "hot".
    - RNN can handle arbitrary input/output lengths.
    - RNN, unlike feedforward neural networks, can use their internal memory to process arbitrary sequences of inputs.
    - RNNs use time-series information (i.e. what I spoke last will impact what I will speak next.)
    - RNNs are ideal for text and speech analysis.

[[1](https://datascience.stackexchange.com/questions/11619/rnn-vs-cnn-at-a-high-level)]
