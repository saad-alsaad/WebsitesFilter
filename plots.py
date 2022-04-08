import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_heatmap(df: pd.DataFrame, image_name: str, color: str, cbar: bool):
    plt.figure(figsize=(130, 130))
    sns.heatmap(df, annot=True, cbar=cbar, fmt="10.2f", cmap=color)
    plt.savefig(f'output/plots/{image_name}.png')
    plt.clf()


def generate_scatter_plot(df: pd.DataFrame, x: str, y: str, image_name: str):
    # plt.figure(figsize=(130, 130))
    sns.scatterplot(data=df, x=x, y=y)
    plt.savefig(f'output/plots/scatter/{image_name}.png')
    plt.clf()


def generate_box_plot(df: pd.DataFrame, x: str, y: str, image_name: str):
    # plt.figure(figsize=(130, 130))
    sns.boxplot(data=df, x=x, y=y)
    plt.savefig(f'output/plots/box/{image_name}.png')
    plt.clf()


def generate_box_plots(df: pd.DataFrame, x: list):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sns.boxplot(data=df, x=x[0], ax=axs[0, 0])
    sns.boxplot(data=df, x=x[1], ax=axs[0, 1])
    sns.boxplot(data=df, x=x[2], ax=axs[1, 0])
    sns.boxplot(data=df, x=x[3], ax=axs[1, 1])
    plt.savefig(f'output/plots/box/{x}.png')
    plt.clf()


def generate_hist_plot(df: pd.DataFrame, x: list, hue: str = 'status'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sns.histplot(data=df, x=x[0], hue=hue, kde=True, ax=axs[0, 0])
    sns.histplot(data=df, x=x[1], hue=hue, kde=True, ax=axs[0, 1])
    sns.histplot(data=df, x=x[2], hue=hue, kde=True, ax=axs[1, 0])
    sns.histplot(data=df, x=x[3], hue=hue, kde=True, ax=axs[1, 1])
    plt.savefig(f'output/plots/hist/{x}.png')
    plt.clf()


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def generate_df_bar_plots(df: pd.DataFrame):
    for column in df.columns:
        values = df[column].value_counts().to_frame().reset_index()
        values.columns = [column, 'count']
        bar_plot = sns.barplot(x=column, y='count', data=values)
        show_values(bar_plot)
        plt.savefig(f'output/plots/bar/{column}_bar_values.png')
        plt.clf()


def bivariate_analysis_relplot(df: pd.DataFrame, x: str, y: str, hue: str):
    sns.relplot(x=x, y=y, hue=hue, data=df)
    plt.savefig(f'output/plots/Bivariate_analysis/{x}_{y}_by_{hue}.png')
    plt.clf()
