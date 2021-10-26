from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from optimizer import PortfolioOptimizer


def plot_performance(model: PortfolioOptimizer, stock_set_number: int, plot_directory: str,
                     figure_size: Tuple[int, int] = (6, 4), label_font_size: int = 12, ticks_font_size: int = 10,
                     legend_font_size: int = 12):
    """
    Generates plot for the portfolios generated over a given set of stocks. Each of the example plots compare the
     expected performance to the realized performance. These plots can be adjusted.

    Args:
        model (PortfolioOptimizer): The model optimized object
        stock_set_number (int): The identifier for the set of stocks that were considered in the optimization
        plot_directory (str): The directory where plots will be stored
        figure_size (Tuple[(int, int)]): The size of the figures that are generated (n inches by m inches)
        label_font_size (int): The font size of all axes labels
        ticks_font_size (int): The font size of all tick labels
        legend_font_size (int): The font size of all legend text

    """

    # Prepare data to plot
    df_to_plot = model.summary.reset_index()
    df_to_plot[['Gamma', 'Average return', 'Risk']] = df_to_plot[['Gamma', 'Average return', 'Risk']].astype(float)

    # Set global font sizes for figures
    plt.rcParams.update({'figure.figsize': figure_size,
                         'axes.labelsize': label_font_size,
                         'xtick.labelsize': ticks_font_size,
                         'ytick.labelsize': ticks_font_size,
                         'legend.fontsize': legend_font_size})

    # Plot pareto frontier (trade-off curve)
    sns.scatterplot(data=df_to_plot, x='Risk', y='Average return', hue='data_class')
    save_plot(f'Pareto frontier set {stock_set_number}', plot_directory, separate_legend=False)

    # Plot return as function of gamma
    sns.lineplot(data=df_to_plot, x='Gamma', y='Average return', hue='data_class')
    save_plot(f'Return set {stock_set_number}', plot_directory, separate_legend=False)

    # Plot risk as function of gamma
    sns.lineplot(data=df_to_plot, x='Gamma', y='Risk', hue='data_class')
    save_plot(f'Risk set {stock_set_number}', plot_directory, separate_legend=False)


def save_plot(plot_label: str, plot_directory: str, legend_cols: int = 2, separate_legend: bool = True):
    """
    Saves the plots in a standardized format.
    Args:
        plot_label (str): The unhyphenated name of the file
        plot_directory (str): The directory where the plot is stored
        legend_cols (int): The number of columns that the legend spans
        separate_legend (bool): Whether or not the legend will be saved as a separate image
    """

    # Replace the spaces in the plot label with hyphens
    hyphenated_label = plot_label.replace(' ', '-')

    # Format the legend
    if separate_legend:  # Separates legend image from plot
        legend = plt.legend(ncol=legend_cols, frameon=False, bbox_to_anchor=(-10, 10), loc='center', borderaxespad=0)
        # Get the legend
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Save separate legend image
        fig.savefig(f'{plot_directory}/{hyphenated_label}-legend.pdf', bbox_inches=bbox)
        legend.remove()  # Removes legend from plot
    else:  # Keeps legend in plot at the lower center
        plt.legend(ncol=legend_cols, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.3))

    # Save plot with minimal white space
    plt.tight_layout()
    plt.savefig(f'{plot_directory}/{hyphenated_label}-plot.pdf')
    plt.show()
