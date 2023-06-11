#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from process_email import process_email
from email_features import email_features
from get_vocabulary_dict import get_vocabulary_dict


def read_file(file_path: str) -> str:
    """Return the content of the text file under the given path.

    :param file_path: path to the file
    :return: file content
    """

    with open(file_path) as text_file: 
        content = text_file.read()

    return content;


# %% ==================== Part 1: Email Preprocessing ====================

print('\nPreprocessing sample email (emailSample1.txt)\n')

file_contents = read_file('data/emailSample1.txt')
word_indices = process_email(file_contents)

# Print Stats
print('Word Indices: \n')
print(word_indices)
print('\n\n')

# input('Program paused. Press enter to continue.\n')

# %% ==================== Part 2: Feature Extraction ====================

print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
file_contents = read_file('data/emailSample1.txt')
word_indices = process_email(file_contents)
features = email_features(word_indices)

# Print Stats
print('Length of feature vector: {}\n'.format(len(features[0])))
print('Number of non-zero entries: {}\n'.format(sum(sum(features > 0))))

# input('Program paused. Press enter to continue.\n')

# %% =========== Part 3: Train Linear SVM for Spam Classification ========

print('\nLoading the training dataset...')
X_train = np.genfromtxt('data/spamTrain_X.csv', delimiter=',')
y_train = np.genfromtxt('data/spamTrain_y.csv', delimiter=',')
print('The training dataset was loaded.')

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

clf = svm.LinearSVC(C=0.1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)

acc_train = clf.score(X_train, y_train)
print('Training Accuracy: {:.2f}%\n'.format(acc_train * 100))

# %% =================== Part 4: Test Spam Classification ================

X_test = np.genfromtxt('data/spamTest_X.csv', delimiter=',')
y_test = np.genfromtxt('data/spamTest_y.csv', delimiter=',')

print('\nEvaluating the trained Linear SVM on a test set ...\n')

y_pred = clf.predict(X_test)

acc_test = clf.score(X_test, y_test)
print('Test Accuracy: {:.2f}%\n'.format(acc_test * 100))

# input('Program paused. Press enter to continue.\n')

# %% ================= Part 5: Top Predictors of Spam ====================

weights = clf.coef_
idx = (- weights).argsort()

vocabulary_dict = get_vocabulary_dict()

print('\nTop predictors of spam: \n')
for i in range(15):
    # FIXME: Replace each `None` with an appropriate expression.
    print(' {word:<20}: {weight:10.6f}'.format(
        word=vocabulary_dict[idx[0,i]+1], weight=weights[0,idx[0,i]]))

print('\n\n')
# input('\nProgram paused. Press enter to continue.\n')

# %% =================== Part 6: Try Your Own Emails =====================

filename = 'data/spamSample2.txt'

# Read and predict
file_contents = read_file(filename)
word_indices = process_email(file_contents)
x = email_features(word_indices)
# FIXME: Predict the labelling.
y_pred = clf.predict(x)

print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename, y_pred[0] > 0))