from libs.RobustSTL import RobustSTL
from libs.crawler import get_data_by_crawling
from libs.utils import load_price_data
from prettytable import PrettyTable

import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
plt.switch_backend('agg')
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

import time 

NOW = time.localtime(time.time())
RUN_TIME = str(NOW.tm_year)
RUN_TIME += '_' + str(NOW.tm_mon)
RUN_TIME += '_' + str(NOW.tm_mday)
RUN_TIME += '_' + str(NOW.tm_hour)
RUN_TIME += '_' + str(NOW.tm_min)

pdf = backend_pdf.PdfPages('./results/STL_analysis_' + RUN_TIME + '.pdf')


def moving_average(x, w=5):
	return np.convolve(x, np.ones(w), 'valid') / w


def gradient(price_data, edge_order=2):
	first_derivative = np.gradient(price_data, edge_order=edge_order)
	return first_derivative


def main(code, name, num_dates):
	input_dir = '/Users/seongokryu/works/finantrics/TrendAnalisys/daily_price/'
	price_data, volume_data = load_price_data(input_dir, code)
	price_data = price_data[-num_dates:].astype(float) 
	volume_data = volume_data[-num_dates:].astype(float) 

	today_price = price_data[-1]
	yesterday_price = price_data[-2]
	price_change = (today_price - yesterday_price) / yesterday_price * 100.0
	mean = np.mean(price_data[:60])
	price_data /= today_price

	result = RobustSTL(price_data, 1, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)

	fig = plt.figure(figsize=(30,25))
	plt.title(code + '\t' + name)
	matplotlib.rcParams.update({'font.size': 22})
	samples = zip(result, ['sample', 'trend', 'seasonality', 'remainder'])

	interval = 10
	n_interval = num_dates // interval
	xticks = np.arange(0, n_interval+1) * interval

	summary = []
	summary.append(code)
	summary.append(name)
	summary.append(today_price)
	summary.append(price_change)

	for i, item in enumerate(samples):
		if i==0:
			plt.subplot(5,1,1)
			plt.title(code + ', ' + name + ', Scaled price')
			plt.plot(item[0], '-o', color='tab:blue', markersize=5)
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)
		elif i==1:    
			continue

		elif i==2:
			plt.subplot(5,1,2)
			plt.plot(item[0]+1.0, '-o', color='tab:orange', markersize=5)
			plt.title('Trend')
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)

			trend = item[0]+1.0
			first_derivative = gradient(trend, 1)
			#second_derivative = gradient(first_derivative, 1)

			plt.subplot(5,1,3)
			plt.plot(first_derivative, '-o', color='tab:red', markersize=5)
			plt.title('First derivative of Trend')
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)
			"""
			plt.subplot(5,1,4)
			plt.plot(second_derivative, '-o', color='tab:cyan', markersize=5)
			plt.title('Second derivative of Trend')
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)
			"""

			summary.append(first_derivative)

		else:
			plt.subplot(5,1,4)
			_x = np.arange(item[0].shape[0])
			_y = np.zeros(item[0].shape[0])
			plt.plot(item[0], '-o', color='tab:green', markersize=5)
			plt.plot(_x, _y, color='black')
			plt.title('Remainder (white noise)')
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)

			plt.subplot(5,1,5)
			_x = np.arange(item[0].shape[0])
			ma = moving_average(volume_data, 5)
			plt.bar(_x, volume_data, color='tab:blue', alpha=0.5)
			plt.plot(_x[4:], ma, color='tab:orange')
			plt.title('Trading volume')
			plt.xticks(xticks)
			plt.xlim(0, num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)

			summary.append(item[0][-1])

	plt.tight_layout()
	plt.savefig('./results/'+code+'_'+name+'.png')
	pdf.savefig(fig)

	return summary


def find_saddle_point(today, yesterday):
	
	if today >= 0.0 and yesterday < 0:
		return 'Up-trend inversion'

	elif today < 0.0 and yesterday > 0.0:
		return 'Down-trend inversion'

	else:
		return '---'


def report_summary(summary_list):

	noise_list = [summary[-1] for summary in summary_list]
	sorted_idx = sorted(range(len(noise_list)), key=lambda k: noise_list[k])

	tab = PrettyTable(['종목코드', '종목명', '종가', '상승률 (%)', 'Noise Level (%)', '오늘 추세 (%)', '어제 추세 (%)'])
	#tab = PrettyTable(['Code', 'Name', 'Today Price', 'Price Change (%)', 'Noise Level (%)', 'Today Trend', 'Yesterday Trend'])
	for idx in sorted_idx:
		summary = summary_list[idx]

		code = summary[0]
		name = summary[1]
		today_price = summary[2]
		price_change = summary[3]
		trend = summary[4]
		noise = summary[5]*100

		today_trend = trend[-1]
		yesterday_trend = trend[-2]

		today_sign = today_trend / np.abs(today_trend)
		today_trend_sign = '(+)'
		if today_sign < 0:
			today_trend_sign = '(-)'

		yesterday_sign = yesterday_trend / np.abs(yesterday_trend)
		yesterday_trend_sign = '(+)'
		if yesterday_sign < 0:
			yesterday_trend_sign = '(-)'

		#tab.add_row([code, name, int(today_price), round(price_change,2), round(noise, 2), today_trend_sign, yesterday_trend_sign])
		tab.add_row([code, name, int(today_price), round(price_change,2), round(noise, 2), round(today_trend*100.0,2), round(yesterday_trend*100.0,2)])

	print (tab)
	print ("Now:", RUN_TIME)
	return


def plot_histogram(summary_list, title, bins=10):

	trend_list = []
	noise_list = []
	change_list = []

	trend_plus = 0
	noise_plus = 0
	change_plus = 0

	trend_minus = 0
	noise_minus = 0
	change_minus = 0

	for summary in summary_list:
		price_change = summary[3]
		trend = summary[4][-1]*100
		noise = summary[5]*100

		trend_list.append(trend)
		noise_list.append(noise)
		change_list.append(price_change)

		if trend > 0.0:
			trend_plus += 1
		else:
			trend_minus += 1

		if noise > 0.0:
			noise_plus += 1
		else:
			noise_minus += 1

		if price_change > 0.0:
			change_plus += 1
		else:
			change_minus += 1

	fig = plt.figure(figsize=(24,6))


	plt.subplot(1,3,1)
	plt.title('(+) 수='+str(change_plus)+', (-) 수='+str(change_minus))
	n, _, _ = plt.hist(change_list, bins=bins, color='tomato', alpha=0.5)
	max_n = np.max(n)
	plt.plot([0,0], [0, max_n], ':', linewidth=2, c='k')
	plt.xlabel('상승률 (%)')
	plt.ylabel('빈도')

	#plt.arrow(0.0, max_n, 1.0, 0.0, head_width=0.05, head_length=0.1, fc='r', ec='r')
	#plt.arrow(0.0, max_n, -1.0, 0.0, head_width=0.05, head_length=0.1, fc='b', ec='b')
	#plt.text(x_min, y_max, '(+) 수:'+str(change_plus), color='r')
	#plt.text(-1.0, max_n-2.0, '(-) 수:'+str(change_minus), color='b')

	plt.subplot(1,3,2)
	plt.title('(+) 수='+str(noise_plus)+', (-) 수='+str(noise_minus))
	n, _, _ = plt.hist(noise_list, bins=bins, color='springgreen', alpha=0.5)
	max_n = np.max(n)
	plt.plot([0,0], [0, max_n], ':', linewidth=2, c='k')
	plt.xlabel('Noise level (%)')
	plt.ylabel('빈도')

	plt.subplot(1,3,3)
	plt.title('(+) 수='+str(trend_plus)+', (-) 수='+str(trend_minus))
	n, _, _ = plt.hist(trend_list, bins=bins, color='royalblue', alpha=0.5)
	max_n = np.max(n)
	plt.plot([0,0], [0, max_n], ':', linewidth=2, c='k')
	plt.xlabel('Trend (%)')
	plt.ylabel('빈도')

	plt.tight_layout()
	plt.savefig('./results/Histogram_'+RUN_TIME+'.png')

	return


if __name__ == '__main__':
	crawling = int(sys.argv[1])
	num_dates = int(sys.argv[2])
	f_name = sys.argv[3]
	list_f = open(f_name+'.txt', 'r')
	stock_list = list_f.readlines()
	print ("Now:", RUN_TIME)

	summary_list = []
	for stock in stock_list:
		code = stock.split(',')[0]
		name = stock.split(',')[1].strip()
		if crawling == 1:
			print (code, name, "start crwaling")
			get_data_by_crawling(code, num_dates)
		summary = main(code, name, num_dates)
		summary_list.append(summary)
	pdf.close()		

	report_summary(summary_list)
	plot_histogram(summary_list, f_name, bins=20)
