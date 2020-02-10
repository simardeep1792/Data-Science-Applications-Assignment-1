# Data-Science-Applications-Assignment-1


Text classification is the process of assigning tags or categories to text according to its content. It’s one of the fundamental tasks in Natural Language Processing (NLP) with applications such as sentiment analysis, topic labeling, spam detection, and intent detection.

# Preparing and Pre-processing the data

In the current assignment, the first step is to create 200 documents with each document containing 150 words and 8 respective authors out of the Gutenberg library. While creating the documents, sentence tokenizer is used to tokenize the sentences out of each book. Once the sentences are tokenized, these are being shuffled up and then loop is initialized where the words are extracted and converted into the documents. The stop words and numerical data are removed to maintain the accuracy and uniformity. In addition, the author name is being appended to the its respective document to keep track of the respective author and its book. 

Once the documents are being made, they are being combined into a single file where all of these are sequentially kept.


# Transforming Data using Bag of Words

For transforming the textual data into the numerical, two of the techniques are used and one of them is Bag of Words. In this model, a text (such as a sentence or a document) is represented as the of bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. This algorithm is used to document the occurrence of each word as a feature and for training a classifier. For generating the bag of words, Count Vectorizer library from sklearn is being used which converts the collection of text documents to a matrix of token counts.

# Training machine with the transformed data using Multinomial NB

The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. Here the data is split into the test and training data and is being fitted with the Multinomial NB algorithm with an accuracy of 70%. By plotting the confusion matrix, the performance of a Multinomial NB model on a set of test data is visualized on the test data for which the true values are known. 


# Predicting any random sentence 

One of the sentences from the book ‘Emma’ by Austen is being chosen and being used to predict against the y-values. The algorithm predicted ‘Austen’ for the book.

# Cross- Validation Evaluation using 10-fold testing

Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model. This method results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.

The general procedure is as follows:
•	Shuffle the dataset randomly.
•	Split the dataset into 10 groups
•	For each unique group, take the group as a hold out or test data set
•	Take the remaining groups as a training data set
•	Fit a model on the training set and evaluate it on the test set

Here at first the 10-fold testing is done without shuffling the sample and the accuracy comes out to be 76%. After that using shuffle split, the whole data is being shuffled and again the 10-fold testing is being performed to get a mean accuracy of 75. At the end, using Standard Scalar, the features are being standardized by removing the mean and scaling to unit variance, that brought down the accuracy to 74%.


# Error Analysis 
During the error analysis, y_test and y_pred_class are being looped to find out the author names which were not predicted correctly by the algorithm. Once those indexes are being found out, then the sentence is being formed out using the X-test. Then that sentence( multiple words) is being used to figure out the author which algorithm predicted and the original author using the vocabulary items.


# Logistic Regression on BOW
Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
In this case, again using the Logistic Regression is being used up to perform analysis on the BOW and performing the 10-fold error analysis on the model with the following results

The objective of a Linear SVC (Support Vector Classifier) is to fit to the data, returning a "best fit" hyperplane that divides, or categorizes, data. From there, after getting the hyperplane, some features are fed to the classifier to see what the "predicted" class is. 
The results with the algorithm are:-
•	SVC Linear BOW: 10 fold test WITHOUT shuffle [0.7875 0.78125 0.7875 0.78125 0.8     0.78125 0.7875  0.7875  0.775 0.7875 ]
•	SVC Linear BOW: Mean Accuracy: 79 %
•	SVC Linear BOW: 10 fold test WITH shuffle [0.73125 0.7375 0.76875 0.7625  0.825   0.80625 0.775   0.8125  0.75625 0.75625]
•	SVC Linear BOW: Mean Accuracy: 77%
•	SVC Linear BOW Standard Scaler: 10-fold test WITH shuffle [0.6     0.68125 0.6625 0.69375 0.70625 0.66875 0.6375  0.64375 0.6 0.6625 ]
•	SVC Linear BOW Standard Scaler: Mean Accuracy 65%




# Transforming Documents to TF-IDF

TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighing factor in searches of information retrieval. The TF-IDF value increases proportionality to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. For creating TF-IDF, TF-IDF vectorizer is being used to convert raw documents to a matrix of TF-IDF features.
