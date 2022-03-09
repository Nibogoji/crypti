import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mplf
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
#
#
# def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
#     for i in net_worth:
#         Date += " {}".format(i)
#     # print(Date)
#     if not os.path.exists('logs'):
#         os.makedirs('logs')
#     file = open("logs/" + filename, 'a+')
#     file.write(Date + "\n")
#     file.close()


class TradingGraph:
    def __init__(self, Render_range):
        self.net_worth = deque(maxlen=Render_range)
        self.pure_investment = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        self.Render_range = Render_range

        self.indicator1 = deque(maxlen=Render_range)
        self.indicator2 = deque(maxlen=Render_range)

        self.net_position = deque(maxlen=Render_range)
        self.var = deque(maxlen=Render_range)

        plt.style.use('ggplot')
        plt.close('all')
        self.fig = plt.figure(figsize=(14, 8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((10, 2), (2, 0), rowspan=5, colspan=1)

        self.ax2 = plt.subplot2grid((10, 2), (9, 0), rowspan=1, colspan=1, sharex=self.ax1)
        self.ax5 = plt.subplot2grid((10, 2), (7, 0), rowspan=2, colspan=1, sharex=self.ax1)
        self.ax4 = plt.subplot2grid((10, 2), (0, 0), rowspan=2, colspan=1, sharex=self.ax1)
        self.ax6 = plt.subplot2grid((10, 2), (2, 1), rowspan=8, colspan=1)

        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.AutoDateFormatter(mpl_dates.AutoDateLocator())

        # plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.8, top=0.88, wspace=0, hspace=0)

    # Render the environment to the screen
    def render(self, Date, Open, High, Low, Close, net_worth, trades,pure_investment, ttd, filter, net_position, step_var, pcs_receipt):

        self.indicator1.append(ttd)
        self.indicator2.append(filter)
        self.net_worth.append(net_worth)
        self.pure_investment.append(pure_investment)
        try:
            self.net_position.append(net_position[-1]['Net position'])

        except:
            self.net_position.append(0)

        self.var.append(step_var)

        Date = mpl_dates.date2num([pd.to_datetime(Date)])[0]
        self.render_data.append([Date, Open, High, Low, Close])

        # Put all dates to one list and fill ax2 sublot with volume
        Date_Render_range = [i[0] for i in self.render_data]###


        self.ax3.clear()
        self.ax3.plot(Date_Render_range, self.net_worth, color="blue", label = 'Trading value')
        # self.ax3.plot(Date_Render_range, self.pure_investment, color="purple", label = 'Pure investment')
        self.ax3.legend()

        self.ax5.clear()
        self.ax5.bar(Date_Render_range, self.var, color="red", label='VAR')
        self.ax5.legend()

        ticks = [0.5*i + 1 for i in range(len(pcs_receipt.columns))]
        self.ax6.clear()
        self.ax6.bar(pcs_receipt.columns, pcs_receipt.values.reshape(-1,))

        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, colorup='green', colordown='red',width=0.8/(24*20), alpha=0.4)

        self.ax2.clear()
        self.ax2.plot(Date_Render_range,self.indicator1, label = 'Train-Test Distance')
        self.ax2.plot(Date_Render_range,self.indicator2,label = 'K2 Corrected Filter')
        self.ax2.legend()
        # self.ax2.fill_between(Date_Render_range, Volume, 0)

        self.ax4.clear()
        self.ax4.plot(Date_Render_range, self.net_position, label='Net Position LTC')
        self.ax4.legend()

        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in Date_Render_range:
                net_position = trade['Net position']
                self.ax4.scatter(trade_date, net_position, c='black', label='black', s=120, edgecolors='none',
                                 marker="^")
                self.ax4.text(trade_date, net_position, str(net_position), fontdict = dict(color='black', alpha = 0.5, size = 16))



        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in Date_Render_range:
                if trade['type'] == 'buy':
                    high_low = trade['Low'] - trade['Low']*0.001
                    self.ax1.scatter(trade_date, high_low, c='yellow', label='yellow', s=120, edgecolors='none',
                                     marker="^")
                    self.ax1.text(trade_date, high_low, 'BUY '+ trade['flag'], fontdict = dict(color='black', alpha = 0.5, size = 16))
                else:
                    high_low = trade['High'] + trade['High']*0.001
                    self.ax1.scatter(trade_date, high_low, c='brown', label='grey', s=120, edgecolors='none', marker="v")
                    self.ax1.text(trade_date, high_low, 'SELL ' + trade['flag'], fontdict = dict(color='black', alpha = 0.5, size = 16))

        self.ax2.set_xlabel('Date')
        self.ax5.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        self.fig.tight_layout()
        self.ax5.axes.get_xaxis().set_visible(True)

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        # plt.show(block=False)
        # Necessary to view frames before they are unrendered
        # plt.pause(0.001)

        """Display image with OpenCV - no interruption"""
        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("AAP Backtrading", image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
