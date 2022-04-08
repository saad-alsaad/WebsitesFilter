import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import threading
from sklearn import preprocessing
import classifiers
import plots
from scipy import stats


def generate_correlations(df: pd.DataFrame, dataset_name: str):
    df_corr = df.corr()
    df_corr.to_csv(f'output/files/{dataset_name}_corr.csv', index=False)
    plots.generate_heatmap(df_corr, f'{dataset_name}_corr', 'YlGnBu', True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    df_out = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
    return df_out


def check_missing_values(df: pd.DataFrame, dataset_name: str):
    df_null = df.isnull()
    df_null.to_csv(f'output/files/{dataset_name}_missing_values.csv', index=False)


def generate_min_max(df_data: pd.DataFrame):
    df_data.min().to_frame().to_csv('min.csv')
    df_data.max().to_frame().to_csv('max.csv')


def generate_summary_statistics(df_data: pd.DataFrame):
    stats_df: pd.DataFrame
    first_quantile_df = df_data.quantile(0.25).to_frame()
    second_quantile_df = df_data.quantile(0.5).to_frame()
    third_quantile_df = df_data.quantile(0.75).to_frame()
    max_df = df_data.max().to_frame()
    min_df = df_data.min().to_frame()
    mean_df = df_data.mean().to_frame()
    mode_df = df_data.mode().reset_index()
    mode_df = pd.melt(mode_df, id_vars=['index'], value_vars=df_data.columns.to_list()).drop('index', axis=1).set_index('variable')
    std_df = df_data.std().to_frame()
    first_quantile_df.columns = ['q1']
    second_quantile_df.columns = ['q2']
    third_quantile_df.columns = ['q3']
    max_df.columns = ['max']
    min_df.columns = ['min']
    mean_df.columns = ['mean']
    mode_df.columns = ['mode']
    std_df.columns = ['Standard Deviation']
    result = pd.concat([first_quantile_df, second_quantile_df, third_quantile_df, max_df, min_df, mean_df, mode_df, std_df], axis=1)
    result.to_csv("output/files/summary_stats.csv", index_label=['feature'])


def normalize_datasets(df: pd.DataFrame, dataset_name: str):
    scaler = preprocessing.MinMaxScaler()
    normalized_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(normalized_array, columns=df.columns)
    scaled_df.to_csv(f'scaled_{dataset_name}.csv', index=False)


def load_original_data() -> dict:
    training_set: pd.DataFrame = pd.read_excel('Data/training_set.xlsx')
    test_set: pd.DataFrame = pd.read_excel('Data/test_set.xlsx')
    # aa = training_set[training_set['path_extension'] > 0]
    # bb = test_set[test_set['path_extension'] > 0]

    # remove constant features
    training_set.drop(['nb_or', 'ratio_nullHyperlinks', 'ratio_intRedirection',
                       'ratio_intErrors', 'submit_email', 'sfh'], axis=1, inplace=True)
    test_set.drop(['nb_or', 'ratio_nullHyperlinks', 'ratio_intRedirection',
                   'ratio_intErrors', 'submit_email', 'sfh'], axis=1, inplace=True)

    training_set['status'].replace({'safe': 1, 'unsafe': 0}, inplace=True)
    test_set['status'].replace({'safe': 1, 'unsafe': 0}, inplace=True)
    training_set['status'] = training_set['status'].astype(int)
    test_set['status'] = test_set['status'].astype(int)

    return {'training_set': training_set, 'test_set': test_set}


def check_high_correlations(df: pd.DataFrame, threshold: float = 0.2):
    set_cor = df.corr()
    cor_target = abs(set_cor['status'])
    relevant_features: pd.Series = cor_target[cor_target > threshold]
    # print(relevant_features.sort_values())
    l1 = relevant_features.index.to_list().copy()
    l2 = relevant_features.index.to_list().copy()
    for ind1 in l1:
        l2.remove(ind1)
        for ind2 in l2:
            cor_value = df[[ind1, ind2]].corr()[ind2].iloc[0]
            if cor_value > threshold:
                print(f'{ind1} and {ind2} correlation is {cor_value}')


def run_classification_algorithms(training_set: pd.DataFrame, test_set: pd.DataFrame):
    x_train = training_set.drop(['status', 'port', 'iframe', 'right_clic'], axis=1)
    y_train = training_set[['status']]
    x_test = test_set.drop(['status', 'port', 'iframe', 'right_clic'], axis=1)
    y_test = test_set[['status']]

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy().ravel()
    print('----------------- Logistic Regression -----------------')
    threads_list = [threading.Thread(target=classifiers.run_logistic_regression, args=(x_train, y_train, x_test, y_test, 10,)),
                    threading.Thread(target=classifiers.run_naive_bayes, args=(x_train, y_train, x_test, y_test, 10,)),
                    threading.Thread(target=classifiers.run_svm, args=(x_train, y_train, x_test, y_test, 10,)),
                    threading.Thread(target=classifiers.run_knn, args=(x_train, y_train, x_test, y_test, 10,)),
                    threading.Thread(target=classifiers.run_random_forest, args=(x_train, y_train, x_test, y_test, 10,))]

    for thrd in threads_list[:2]:
        thrd.start()

    for thrd in threads_list[:2]:
        thrd.join()

    for thrd in threads_list[2:]:
        thrd.start()

    for thrd in threads_list[2:]:
        thrd.join()


if __name__ == '__main__':

    original_datasets = load_original_data()
    generate_summary_statistics(original_datasets['training_set'])

    plots.generate_box_plots(original_datasets['training_set'], ['google_index', 'page_rank', 'nb_www', 'nb_hyperlinks'])
    training_set: pd.DataFrame = pd.read_csv('scaled_training_set.csv')
    test_set: pd.DataFrame = pd.read_csv('scaled_test_set.csv')

    training_set['status'].replace({1.0: 'safe', 0.0: 'unsafe'}, inplace=True)

    plots.generate_scatter_plot(training_set[['page_rank', 'status']], x='page_rank', y='status',
                          image_name='page_rankVSstatus_scatter')

    plots.generate_box_plot(training_set, x='status', y='page_rank', image_name='page_rankVSstatus_box')

    plots.bivariate_analysis_relplot(training_set, x='page_rank', y='domain_age', hue='status')

    normalize_datasets(training_set, 'training_set')
    normalize_datasets(test_set, 'test_set')
    generate_min_max(training_set)

    plots.generate_hist_plot(training_set, ['longest_word_path', 'longest_words_raw', 'longest_word_host', 'ratio_digits_url'])

    run_classification_algorithms(training_set, test_set)

