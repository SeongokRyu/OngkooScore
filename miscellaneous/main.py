from libs.RobustSTL import RobustSTL
from libs.crawler import get_data_by_crawling
from libs.utils import load_price_data

import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
plt.switch_backend('agg')

import time 

NOW = time.localtime(time.time())
RUN_TIME = str(NOW.tm_year)
RUN_TIME += '_' + str(NOW.tm_mon)
RUN_TIME += '_' + str(NOW.tm_mday)
RUN_TIME += '_' + str(NOW.tm_hour)
RUN_TIME += '_' + str(NOW.tm_min)

pdf = backend_pdf.PdfPages('./results/STL_analysis_' + RUN_TIME + '.pdf')

def gradient(price_data, edge_order=2):
	first_derivative = np.gradient(price_data, edge_order=edge_order)
	return first_derivative

def main(code, name, num_dates):
	input_dir = '/Users/seongokryu/works/finantrics/TrendAnalisys/daily_price/'
	price_data = load_price_data(input_dir, code)
	price_data = price_data[-num_dates:].astype(float) 
	today_price = price_data[-1]
	price_data /= today_price

	result = RobustSTL(price_data, 1, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)

	fig = plt.figure(figsize=(30,25))
	plt.title(code + '\t' + name)
	matplotlib.rcParams.update({'font.size': 22})
	samples = zip(result, ['sample', 'trend', 'seasonality', 'remainder'])

	interval = 30
	n_interval = num_dates // interval
	xticks = np.arange(0, n_interval+1) * interval

	for i, item in enumerate(samples):
		if i==0:
			plt.subplot(3,1,1)
			plt.title(code + ', ' + name + ', Scaled price')
			plt.plot(item[0], '-o', color='blue', markersize=5)
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)
		elif i==1:    
			continue
		elif i==2:
			plt.subplot(3,1,i)
			plt.plot(item[0]+1.0, '-o', color='red', markersize=5)
			plt.title('Trend')
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)
			first_derivative = gradient(item[0]+1.0, 1)
			print (price_data[-20:])
			print (first_derivative[-20:])
			exit(-1)
		else:
			plt.subplot(3,1,i)
			_x = np.arange(item[0].shape[0])
			_y = np.zeros(item[0].shape[0])
			plt.plot(item[0], '-o', color='green', markersize=5)
			plt.plot(_x, _y, color='black')
			plt.title('Remainder (white noise)')
			plt.xticks(xticks)
			plt.xlim(0,num_dates)
			plt.grid(True, linestyle='dashed', linewidth=0.5)

	plt.savefig('./results/'+code+'_'+name+'.png')
	pdf.savefig(fig)

	noise = result[-1][-1]
	summary = [
		code,
		name,
		today_price,
		noise
	]

	return summary

def report_summary(summary_list):

	# Report noise case 
	very_low_noise = []
	low_noise = []
	high_noise = []
	very_high_noise = []

	for idx, summary in enumerate(summary_list):
		noise = summary[-1]
		if noise <= -0.05:
			very_low_noise.append(idx)

		elif noise > -0.05 and noise <= 0.0:
			low_noise.append(idx)

		elif noise > 0.0 and noise <= 0.05:
			high_noise.append(idx)

		elif noise > 0.05:
			very_high_noise.append(idx)

	print ("Extremely low noise stocks")
	for idx in very_low_noise:
		summary = summary_list[idx]
		code = summary[0]
		name = summary[1]
		today_price = summary[2]
		noise = summary[3]*100
		print (code, "\t", name, "\t Today price:", int(today_price), "\t Noise-level:", round(noise,3), "%")

	print ("\n")
	print ("Low noise stocks")
	for idx in low_noise:
		summary = summary_list[idx]
		code = summary[0]
		name = summary[1]
		today_price = summary[2]
		noise = summary[3]*100
		print (code, "\t", name, "\t Today price:", int(today_price), "\t Noise-level:", round(noise,3), "%")

	print ("\n")
	print ("High noise stocks")
	for idx in high_noise:
		summary = summary_list[idx]
		code = summary[0]
		name = summary[1]
		today_price = summary[2]
		noise = summary[3]*100
		print (code, "\t", name, "\t Today price:", int(today_price), "\t Noise-level:", round(noise,3), "%")

	print ("\n")
	print ("Extremely high noise stocks")
	for idx in very_high_noise:
		summary = summary_list[idx]
		code = summary[0]
		name = summary[1]
		today_price = summary[2]
		noise = summary[3]*100
		print (code, "\t", name, "\t Today price:", int(today_price), "\t Noise-level:", round(noise,3), "%")

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
