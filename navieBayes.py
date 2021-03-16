import csv
import numpy as np
import random
import warnings
warnings.simplefilter("ignore")

def preprocess(filename):
    spam = []
    notSpam = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if int(row[len(row) - 1]) == 1:
                spam.append(row[:-1])
            else:
                notSpam.append(row[:-1])
    return spam, notSpam

# Obtain a probabilitic model
def train(train_spam, train_notSpam):
    mean_train_spam = np.mean(train_spam, dtype=np.float64, axis=0)
    std_train_spam = np.std(train_spam, dtype=np.float64, axis=0)
    std_train_spam = np.where(std_train_spam == 0, 0.0001, std_train_spam)
    mean_train_notSpam = np.mean(train_notSpam, dtype=np.float64, axis=0)
    std_train_notSpam = np.std(train_notSpam, dtype=np.float64, axis=0)
    std_train_notSpam = np.where(std_train_notSpam == 0, 0.0001, std_train_notSpam)
    return mean_train_spam, mean_train_notSpam, std_train_spam, std_train_notSpam

# run probabilist model on test set
def test(prior_class1, prior_class0, mean_train_spam, mean_train_notSpam, std_train_spam, std_train_notSpam, test_spam,
         test_notSpam):
    act_spam = []
    for i in range(len(test_spam)):
        # print(i)
        act_1 = np.log(prior_class1)
        temp = ((test_spam[i] - mean_train_spam) ** 2) / (2 * std_train_spam * std_train_spam)
        temp = np.log((1 / (np.sqrt(2 * np.pi) * std_train_spam)) * np.exp(-temp))
        act_1 += np.sum(temp)    # probability the test example is in the spam class
        act_0 = np.log(prior_class0)
        temp = ((test_spam[i] - mean_train_notSpam) ** 2) / (2 * std_train_notSpam * std_train_notSpam)
        temp = np.log((1 / (np.sqrt(2 * np.pi) * std_train_notSpam)) * np.exp(-temp))
        act_0 += np.sum(temp)   # probability the test example is in the not spam class
        if act_1 == max(act_1, act_0):   # determine the classification, max between prob of spam and prob of not spam
            act_spam.append(1)
        else:
            act_spam.append(0)
    act_notSpam = []
    for i in range(len(test_notSpam)):
        act_1 = np.log(prior_class1)
        temp = ((test_notSpam[i] - mean_train_spam) ** 2) / (2 * std_train_spam * std_train_spam)
        temp = np.log((1 / (np.sqrt(2 * np.pi) * std_train_spam)) * np.exp(-temp))
        act_1 += np.sum(temp)     # probability the test example is in the spam class
        act_0 = np.log(prior_class0)
        temp = ((test_notSpam[i] - mean_train_notSpam) ** 2) / (2 * std_train_notSpam * std_train_notSpam)
        temp = np.log((1 / (np.sqrt(2 * np.pi) * std_train_notSpam)) * np.exp(-temp))
        act_0 += np.sum(temp)    # probability the test example is in the not spam class
        if act_1 == max(act_1, act_0):  # determine the classification, max between prob of spam and prob of not spam
            act_notSpam.append(1)
        else:
            act_notSpam.append(0)
    return act_spam, act_notSpam


def compute_accuracy_metrics(act_spam, act_notSpam):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for a in act_spam:
        if a == 1:
            tp += 1
        else:
            fn += 1
    for a in act_notSpam:
        if a == 0:
            tn += 1
        else:
            fp += 1
    accuracy = ((tp + tn) / (tp + fp + tn + fn)) * 100
    precision = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100
    print("Confusion Matrix")
    print(tp, fp)
    print(fn, tn)
    return accuracy, precision, recall


# divy intp training and testing sets
spam, notSpam = preprocess("spambase.data")
random.shuffle(spam)
random.shuffle(notSpam)
test_spam = spam[:len(spam) // 2]
test_notSpam = notSpam[:len(notSpam) // 2]
train_spam = spam[len(spam) // 2:]
train_notSpam = notSpam[len(notSpam) // 2:]

# Calculate the  prior probabilities on the training set
pb_prior_class1 = len(train_spam) / (len(train_spam) + len(train_notSpam))
pb_prior_class0 = len(train_notSpam) / (len(train_spam) + len(train_notSpam))

train_spam = np.array(train_spam, dtype=np.float64)
train_notSpam = np.array(train_notSpam, dtype=np.float64)
test_spam = np.array(test_spam, dtype=np.float64)
test_notSpam = np.array(test_notSpam, dtype=np.float64)

# Probability Model on training set
mean_train_spam, mean_train_notSpam, std_train_spam, std_train_notSpam = train(train_spam, train_notSpam)

# Predicted classes on testing set
act_spam, act_notSpam = test(pb_prior_class1, pb_prior_class0, mean_train_spam, mean_train_notSpam, std_train_spam,
                             std_train_notSpam, test_spam, test_notSpam)

# compute acccuracy metrics
accuracy, precision, recall = compute_accuracy_metrics(act_spam, act_notSpam)

print(accuracy, precision, recall)
print(pb_prior_class0, pb_prior_class1)
