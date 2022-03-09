import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import ticker
import matplotlib.pyplot as plt
import mplfinance as mpf

"""
CONTAINS : CustomLocator, plt2grid, ohlc_plotly, ohlc_mpf

"""
class CustomLocator(ticker.MaxNLocator):
    def ticks_values(self, vmin, vmax):
        if len(np.arange(vmin,vmax,1)) > 20:
            tcks = list(np.linspace(vmin,vmax,num=20))
        else:
            tcks = list(np.linspace(vmin,vmax,num=len(np.arange(vmin,vmax,1))))

        return tcks

# data_dict_ex = {'ax1': {'0': {'data': [],
#                               'label': [],
#                               'color': []},
#                        '1': {'data': [],
#                               'label': [],
#                               'color': []},
#                        '2': {'data': [],
#                               'label': [],
#                               'color': []}},
#                 'ax2':{},
#                 'ax3':{}
# }
#
# plotOrbar_ex = {'ax1': {'0': 'plot',
#                        '1': 'plot',
#                        '2': 'plot'},
#                 'ax2': {'0': 'bar',
#                        '1': 'bar'},
#                 'ax3': {}
# }


def plt2grid(data, subplot_slots, subplot_start, share_ax, plotOrbar, fig_n = 1000, show = True, ML = CustomLocator()):

    """
    Subplot2grid

    :param data: dictionary of data to plot. First lvl keys ax1,ax2,..., axn.
                                            Second lvl keys: range of series to plot per ax i
                                            Third lvl keys: data, label, color
    :param subplot_slots: list of Subplots dimension
    :param subplot_start: list of Subplots starting position
    :param share_ax: list of False/ axis
    :param plotOrbar: dict with 'plot' or 'bar' for each series for each ax
    :param fig_n: figure number
    :return: figure
    """

    n_subplots = len(data.keys())

    fig = plt.figure(fig_n)

    ax = {}
    for sp in reversed(range(n_subplots)):

        if sp != max([i for i in range(n_subplots)]):

            ax['ax{}'.format(sp+1)] = plt.subplot2grid((sum(subplot_slots), 1),
                                                     (subplot_start[sp], 0),
                                                     rowspan=subplot_slots[sp],
                                                     colspan=1,
                                                     sharex = ax[share_ax[-1]])
        else:
            ax['ax{}'.format(sp + 1)] = plt.subplot2grid((sum(subplot_slots), 1),
                                                             (subplot_start[sp], 0),
                                                             rowspan=subplot_slots[sp],
                                                             colspan=1)

    for k in ax.keys():
        for i,v in enumerate(data[k].keys()):

            if plotOrbar[k][v] == 'plot':

                ax[k].plot(list(data[k][v]['data'].index.astype(str)),
                       data[k][v]['data'],
                       label=data[k][v]['label'],
                       color=data[k][v]['color'])

            elif plotOrbar[k][v] == 'bar':

                ax[k].bar(list(data[k][v]['data'].index.astype(str)),
                       data[k][v]['data'],
                       label=data[k][v]['label'],
                       color=data[k][v]['color'])

            else:
                print('Plot or Bar not specified')

    for i,k in enumerate(ax.keys()):

        if i == 0:

            ax[k].xaxis.set_visible(True)
            ax[k].xaxis.set_major_locator(ML)
            ax[k].legend()

        else:
            ax[k].xaxis.set_visible(False)
            ax[k].xaxis.set_major_locator(ML)
            ax[k].legend()

    fig.autofmt_xdate()

    if show:

        fig.show()

    return fig


def ohlc_plotly(ohlc, show = True):

    fig = go.Figure(data=go.Ohlc(x=ohlc.index,
                    open =  ohlc['Price daily product']['open'],
                    close = ohlc['Price daily product']['close'],
                    high =  ohlc['Price daily product']['high'],
                    low = ohlc['Price daily product']['low'],
                    ))

    if show:
        fig.show(renderer = 'png')

    return fig


def ohlc_mpf(ohlc,title='',mav = False):

    """
    :param ohlc: Dataframe with columns = ['Open','High','Low','Close']

    """
    if mav:

        mpf.plot(ohlc,type = 'candle',title = title, mav = mav)

    else:

        mpf.plot(ohlc, type='candle', title=title)


def ohlc_plot_weekly(dictionary, year=2018, week=15,mav = False):

    first_key = 'products_{}'.format(str(year))
    second_key = str(year)[-2:] + '-{}'.format(week)
    data = dictionary[first_key][second_key]
    data.dropna(inplace = True)
    ohlc_mpf(data,title = second_key,mav=mav)









#   SINGLE PLOT STRUCTURE
#
# from matplotlib import ticker
#
#
# class CustomLocator(ticker.MaxNLocator):
#     def ticks_values(self, vmin, vmax):
#         if len(np.arange(vmin,vmax,1)) > 20:
#             tcks = list(np.linspace(vmin,vmax,num=20))
#         else:
#             tcks = list(np.linspace(vmin,vmax,num=len(np.arange(vmin,vmax,1))))
#
#         return tcks
#
# ML = CustomLocator()
#
# fig = plt.figure(111)
#
# ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
# ax1 = plt.subplot2grid((4,1), (1,0),  rowspan=2, colspan=1, sharex=ax2)
# ax3 = plt.subplot2grid((4,1), (0,0),  rowspan=1, colspan=1, sharex=ax2)
#
#
# ax1.plot(list(complete_df.index.astype(str)),complete_df['price buy'], label = 'buy', color = 'orange')
# ax1.plot(list(complete_df.index.astype(str)),complete_df['price sell'], label = 'Sell',color = 'blue')
#
# ax2.plot(list(complete_df.index.astype(str)),complete_df['volume buy'], label = 'Volume buy',color = 'orange')
# ax2.plot(list(complete_df.index.astype(str)),complete_df['volume sell'], label = 'Volume sell',color = 'blue')
#
# ax3.plot(list(complete_df.index.astype(str)), complete_df['time to delivery'],label = 'Time to delivery',color = 'purple')
#
# ax1.xaxis.set_visible(False)
# ax3.xaxis.set_visible(False)
#
#
# ax2.xaxis.set_major_locator(ML)
#
# fig.autofmt_xdate()
#
# ax1.legend()
# ax2.legend()
# ax3.legend()
#
# fig.show()