from typing import IO
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, explained_variance_score, \
    mean_squared_error, mean_absolute_error, accuracy_score, r2_score, SCORERS
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB, ComplementNB
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, KNeighborsTransformer
from sklearn.model_selection import cross_val_score, learning_curve
from mlxtend.evaluate import bias_variance_decomp


def print_results(y_test: pd.DataFrame, y_pred: pd.DataFrame, x_train: pd.DataFrame, y_train: pd.DataFrame,
                  x_test: pd.DataFrame, train_y_pred: pd.DataFrame, cls_model, cross_fold: int, file: IO, k: int = 3):
    scoring_list = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error',
                    'accuracy', 'balanced_accuracy', 'average_precision', 'precision', 'recall', 'f1']
    train_sizes = [1, 100, 500, 2000, 5000, 7201]
    print(f'---------- Start with cross validation fold = {cross_fold} -------------')
    for score in scoring_list:
        try:
            cr_val_score: np.ndarray = cross_val_score(cls_model, x_train, y_train, cv=cross_fold, scoring=score)
            if score in ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error'
                    , 'neg_root_mean_squared_error']:
                cr_val_score = -cr_val_score
            file.write(f'Cross Validation {score} Scores for {cross_fold} fold: {cr_val_score.round(4)}\n')
            file.write(f'Cross Validation Mean {score} Score for {cross_fold} fold: {cr_val_score.mean()}\n')
        except Exception as e:
            file.write(f'Exception happened when calling cross_val_score, args: cross_fold: {cross_fold}, score: {score} - Msg: {e}\n')

    print(f'----------- Get Training and Validation Errors -------------')
    if cls_model.__class__.__name__ == 'LogisticRegression':
        estimator_class = cls_model.__class__(max_iter=200)
    elif cls_model.__class__.__name__ == 'RandomForestClassifier':
        estimator_class = cls_model.__class__(oob_score=True)
    elif cls_model.__class__.__name__ == 'KNeighborsClassifier':
        estimator_class = cls_model.__class__(n_neighbors=k)
    else:
        estimator_class = cls_model.__class__()

    train_sizes, train_scores, validation_scores = learning_curve(estimator=estimator_class, X=x_train, y=y_train,
                                                                  train_sizes=train_sizes, cv=cross_fold,
                                                                  scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(f'Learning curves for {cls_model.__class__.__name__} model', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f'output/plots/training_val_errors/{cls_model.__class__.__name__}.png')
    plt.clf()

    print(f'---------- Start with calculating bias and variance -------------')
    try:
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            cls_model, x_train, y_train, x_test, y_test)

        file.write('Average expected loss: %.3f\n' % avg_expected_loss)
        file.write('Average bias: %.3f\n' % avg_bias)
        file.write(f'Average variance: %.3f\n' % avg_var)
    except Exception as err:
        file.write(f'Exception happened when calling bias_variance_decomp - Msg: {err}\n')

    ac = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    training_conf_matrix = confusion_matrix(y_train, train_y_pred)
    report = classification_report(y_test, y_pred)
    variance = explained_variance_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    training_mse = mean_squared_error(y_train, train_y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    file.write(f'Variance: {variance}\n')
    file.write(f'Train Scores Mean: {train_scores_mean}\n')
    file.write(f'Validation Scores Mean: {validation_scores_mean}\n')
    file.write(f'MSE: {mse}\n')
    file.write(f'Training MSE: {training_mse}\n')
    file.write(f'MAE: {mae}\n')
    file.write(f'Accuracy Score: {ac}\n')
    file.write(f'Confusion Matrix: {conf_matrix}\n\n')
    file.write(f'Training Confusion Matrix: {training_conf_matrix}\n\n')
    file.write(f'Report: {report}\n\n')


def run_logistic_regression(x_train: pd.DataFrame, y_train: pd.DataFrame,
                            x_test: pd.DataFrame, y_test: pd.DataFrame, cross_fold: int):

    logistic_reg = LogisticRegression(max_iter=200)
    logistic_reg.fit(x_train, y_train)
    y_pred = logistic_reg.predict(x_test)
    train_y_pred = logistic_reg.predict(x_train)

    with open(f'{logistic_reg.__class__.__name__}_log.txt', 'w') as file:
        print_results(y_test, y_pred, x_train, y_train, x_test, train_y_pred, logistic_reg, cross_fold, file)


def run_random_forest(x_train: pd.DataFrame, y_train: pd.DataFrame,
            x_test: pd.DataFrame, y_test: pd.DataFrame, cross_fold: int):

    clf = RandomForestClassifier(oob_score=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    train_y_pred = clf.predict(x_train)
    with open(f'{clf.__class__.__name__}_log.txt', 'w') as file:
        print_results(y_test, y_pred, x_train, y_train, x_test, train_y_pred, clf, cross_fold, file)


def run_naive_bayes(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame,
                    cross_fold: int):

    classifiers = [GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB, ComplementNB]
    for naive_bayes_cls in classifiers:
        classifier = naive_bayes_cls()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        train_y_pred = classifier.predict(x_train)

        with open(f'{classifier.__class__.__name__}_log.txt', 'w') as file:
            print_results(y_test, y_pred, x_train, y_train, x_test, train_y_pred, classifier, cross_fold, file)


def run_svm(x_train: pd.DataFrame, y_train: pd.DataFrame,
            x_test: pd.DataFrame, y_test: pd.DataFrame, cross_fold: int):

    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    train_y_pred = clf.predict(x_train)
    with open(f'{clf.__class__.__name__}_log.txt', 'w') as file:
        print_results(y_test, y_pred, x_train, y_train, x_test, train_y_pred, clf, cross_fold, file)


def run_knn(x_train: pd.DataFrame, y_train: pd.DataFrame,
            x_test: pd.DataFrame, y_test: pd.DataFrame, cross_fold: int):
    max_accuracy, best_k = 0, 0
    best_y_pred, best_clf, best_train_y_pred, best_x_train, best_y_train, best_x_test, \
    best_y_test = None, None, None, None, None, None, None

    for k in range(1, 31):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        train_y_pred = clf.predict(x_train)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k
            best_y_pred = y_pred
            best_clf = clf
            best_train_y_pred = train_y_pred
            best_x_train = x_train
            best_y_train = y_train
            best_x_test = x_test
            best_y_test = y_test

    with open(f'{best_clf.__class__.__name__}_log.txt', 'w') as file:
        file.write(f"Best K: {best_k}\n")
        print_results(best_y_test, best_y_pred, best_x_train, best_y_train, best_x_test, best_train_y_pred, best_clf, cross_fold, file, k=best_k)
