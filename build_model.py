# Importing standard libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Importing Libraries for Modeling
# import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import preprocessing as p

"""
Feature importance testing with RandomForestClassifier
Insignificant feature dropping
Model building with KNearest Neighbors
"""


got_clean = p.clean_variables()
#############################################################################
## Model development
#############################################################################

## Feature importance testing
# checking which features DONT influence the target variable
got_data = got_clean.loc[:, ~got_clean.columns.isin(['isAlive'])]
got_target = got_clean.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(got_data, got_target, test_size=0.1, random_state=508, stratify=got_target)

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators=1100,
                                          criterion='gini',
                                          max_depth=None,
                                          min_samples_leaf=16,
                                          bootstrap=False,
                                          warm_start=True,
                                          random_state=508)

# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


# Feature importance

def plot_feature_importances(model, train=X_train, export=False):
    fig, ax = plt.subplots(figsize=(12, 9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


plot_feature_importances(full_gini_fit,
                         train=X_train,
                         export=False)

#############################################################################
# dropping insignificant features

got_clean = got_clean.drop(columns=['isNoble',
                                    'Title_U',
                                    'House_U',
                                    'isMarried',
                                    'Knight',
                                    'Lady',
                                    'Lord',
                                    'Maester',
                                    'House_Reach',
                                    'Riverlands',
                                    'Vale',
                                    'Septas',
                                    'King_Prince',
                                    'Dothraki',
                                    'Ghiscari',
                                    'Free_Folk',
                                    'House_Dorne',
                                    'House_Vale',
                                    'Dornish',
                                    'Cult_U',
                                    'House_Westerlands',
                                    'House_Riverlands'
                                    ])

## test split

got_data = got_clean.loc[:, ~got_clean.columns.isin(['isAlive'])]
got_target = got_clean.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
    got_data,
    got_target,
    test_size=0.1,
    random_state=508,
    stratify=got_target)

############################################################################
# KNN - using KNearest Neighbors to build a predictive model

# Running the neighbor optimization code with a small adjustment for classification

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train.values.ravel())

    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(12, 9))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Looking for the highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

# It looks like 4 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors=17)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

knn_clf_fit = knn_clf.fit(X_train, y_train.values.ravel())

# Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

# Generating Predictions based on the optimal KNN model
knn_clf_pred = knn_clf_fit.predict(X_test)

# Sending the Predictions to an Excel file
model_predictions_df = pd.DataFrame({'GOT_Actual': y_test,
                                     'GOT_Predicted': knn_clf_pred})

model_predictions_df.to_excel(r"predictions/GOT_predictions.xlsx")

############################################################################
# Validation and confusion matrix
############################################################################

############################################################################
# Creating a confusion matrix
print(confusion_matrix(y_true=y_test,
                       y_pred=knn_clf_pred))

# Visualizing a confusion matrix

labels = ['Dead', 'Alive']

cm = confusion_matrix(y_true=y_test,
                      y_pred=knn_clf_pred)

sns.heatmap(cm,
            annot=True,
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

############################################################################
# Creating a classification report

from sklearn.metrics import classification_report

print(classification_report(y_true=y_test,
                            y_pred=knn_clf_pred))

# Changing the labels on the classification report
print(classification_report(y_true=y_test,
                            y_pred=knn_clf_pred,
                            target_names=labels))

############################################################################
# Cross Validation with k-folds

# Cross Validating the knn model with three folds
cv_knn_3 = cross_val_score(knn_clf,
                           got_data,
                           got_target,
                           cv=3)

print(cv_knn_3)

print(np.mean(cv_knn_3).round(3))

print('\nAverage: ',
      np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))

# Cross Validating the knn model with three folds
cv_knn_3 = cross_val_score(knn_clf,
                           got_data,
                           got_target,
                           cv=3)

print(cv_knn_3)

print(np.mean(cv_knn_3).round(3))

print('\nAverage: ',
      np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))
