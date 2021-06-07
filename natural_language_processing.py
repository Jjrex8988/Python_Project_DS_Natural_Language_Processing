# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)


# Cleaning the texts
import re
import nltk


# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)
print('-' * 38)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(max_features=1500)
print(cv)
print('-' * 38)

X = cv.fit_transform(corpus).toarray()
print(X)
print('-' * 38)

y = dataset.iloc[:, -1].values
print(y)
print('-' * 38)


print(len(X[0]))
print('-' * 38)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print(X_train)
print('-' * 38)
print(X_test)
print('-' * 38)
print(y_train)
print('-' * 38)
print(y_test)
print('-' * 38)



# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)


## LR
# from sklearn.linear_model import LogisticRegression
#
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)


## K-NN
# from sklearn.neighbors import KNeighborsClassifier
#
# classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# classifier.fit(X_train, y_train)


## SVM
# from sklearn.svm import SVC
#
# classifier = SVC(kernel='linear', random_state=0)
# classifier.fit(X_train, y_train)


## KERNEL SVM
# from sklearn.svm import SVC
#
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)


## DT
# from sklearn.tree import DecisionTreeClassifier
#
# classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
# classifier.fit(X_train, y_train)


## RF
# from sklearn.ensemble import RandomForestClassifier
#
# classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
# classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
print('-' * 38)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
print('-' * 38)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print('-' * 38)


## POSITIVE
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
print('-' * 38)


## NEGATIVE
new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)