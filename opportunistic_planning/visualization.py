import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime


def plot_dimensions(results_df, save=False, rnn=None):
    # TODO: adapt to only consider given dimensions (e.g. only xy, xyz)
    
    parameter_c = [float(x[3:6]) for x in results_df.columns.tolist() \
                   if x != 'sequence' and x != 'ID' and x != 'error']
        
    parameter_k = [float(x[11:14]) for x in results_df.columns.tolist() \
                   if x != 'sequence' and x != 'ID' and x != 'error']
    
    dimensions = [x.strip() for x in results_df.columns[:-3].str.split(';').str[2]]
    median = [x for x in results_df.loc['median'].tolist() if pd.notnull(x)]
    colors = ['#e74c3c', '#0064a3', '#70b85d', '#287d78', '#54d0ff', '#f1e664', '#fd8f00']
    dimension_number = [0 if x=='x' else 1 if x=='y' else 2 if x=='z' \
                        else 3 if x=='xy' else 4 if x=='xz' \
                        else 5 if x=='yz' else 6 for x in dimensions]
        
    color_map = matplotlib.colors.ListedColormap(colors)
    ticks = ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']
    
    fig = plt.figure(figsize=(24,17))
    
    # add 3d background, set to white
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
    ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
    ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
    
    # plot results
    img = ax.scatter(parameter_c, parameter_k, median, alpha=0.5, s=38,
                     c=dimension_number, cmap=color_map)
    
    # plot rnn
    if rnn != None:
        xx, yy = np.meshgrid(np.linspace(1,1.9), np.linspace(0,0.9))
        zz = xx * 0 + np.mean(rnn)
        ax.plot_surface(xx, yy, zz, alpha=0.5)
    
    # set labels
    ax.set_ylabel('parameter k', fontsize=18, labelpad=10)
    ax.set_xlabel('parameter c', fontsize=18, labelpad=10)
    ax.set_zlabel('accumulated prequential error', fontsize=18, labelpad=7)
    
    # add colorbar
    cb = plt.colorbar(img, cax=fig.add_axes([0.82,0.3,0.03,0.4]))
    cb.set_alpha(0.8)
    cb.draw_all()
    cb.ax.set_yticklabels(ticks, fontsize=16)
    
    # set plot orientation
    ax.azim = -40
    ax.dist = 10
    ax.elev = 10
    
    plt.margins(0, 0.01)
    
    if save == True:
        date = datetime.today().strftime('%Y-%m-%d')
        filename = 'plot_dimensions_' + str(date) + '.png'
        plt.savefig(filename, bbox_inches='tight', dpi=600)
    
    plt.show()


def plot_comparison_to_baselines(results_median, lowest_mean_idx, lowest_median,
                                 save=False, cpt=None, rnn=None):
    
    IDs = results_median['ID'][:-2]
    sequences = results_median['sequence'][:-2].values
    
    #lowest_mean, lowest_mean_idx, lowest_median, results_df = processing.get_lowest_error(results_median)
    
    results = results_median[lowest_mean_idx[0]][:-2].values
    median = [np.nanmedian(results)] * len(results)
    x = [x for x in range(0, len(sequences))]
    
    
    plt.figure(figsize=(24,16))
    
    # plot results
    plt.scatter(x, results, marker='o', s=26, color='darkviolet', alpha=0.8,
                label=str('model-generated median: ') + str(round(lowest_median,3)))
    plt.plot(x, median, color='darkviolet', alpha=0.95, linewidth=2)
    
    # plot cpt
    if cpt != None:
        plt.scatter(x, cpt, marker='o', s=20, color='dodgerblue', alpha=0.3,
                    label=str('CPT baseline median: ') + str(round(np.median(cpt),3)))
        plt.plot(x, [np.median(cpt)] * len(x), '-', color='dodgerblue', alpha=0.9, linewidth=2)
        plt.fill_between(x, cpt, alpha=0.3, color='dodgerblue')
    
    # plot rnn
    if rnn != None:
        plt.scatter(x, rnn, marker='o', s=20, color='limegreen', alpha=0.5,
                    label=str('RNN baseline median: ') + str(round(np.median(rnn),3)))
        plt.plot(x, [np.median(rnn)] * len(x), '-', c='green', alpha=0.9, linewidth=2)
        plt.fill_between(x, rnn, alpha=0.3, color='limegreen')
        
    plt.xticks(x, labels=IDs, rotation=90, fontsize=5)
    plt.ylabel('accumulated prediction error', fontsize=22)
    plt.xlabel('sequence', fontsize=22)
    plt.margins(0.01)
    
    plt.legend(fontsize=20, framealpha=0.8, loc='upper right', markerscale=2.5)
    
    if save == True:
        date = datetime.today().strftime('%Y-%m-%d')
        filename = 'plot_comparison_ml_baselines_' + str(date) + '.png'
        plt.savefig(filename, bbox_inches='tight', dpi=600)
    
    plt.show()
    
    
    
