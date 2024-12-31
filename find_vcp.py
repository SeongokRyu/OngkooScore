import time

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

import matplotlib
import matplotlib.pyplot as plt


def get_price(
		code,
		start,
		end
	):
	df = fdr.DataReader(code, start, end)
	return df


def gradient(price_data, edge_order=2):
	first_derivative = np.gradient(price_data, edge_order=edge_order)
	return first_derivative


def edit_(price_list, period=200, n=20):
	ma_list = []
	vol_list = []
	for i in range(period):
		start = 240-(200-i-1)-n
		end = 240-(200-i-1)
		sliced = price_list[start:end]
		vol = np.std(sliced)
		ma = np.mean(sliced)

		vol_list.append(vol)
		ma_list.append(ma)
	return price_list[-200:], vol_list, ma_list


def plot2(
		code,
		price_list,
		ma_list,
		vol_list,
	):
	vol_dev = gradient(vol_list)
	ma_dev = gradient(ma_list)
	
	fig = plt.figure(figsize=(30,25))
	matplotlib.rcParams.update({'font.size': 22})

	num_dates = price_list.shape[0]
	interval = 10
	n_interval = num_dates // interval
	xticks = np.arange(0, n_interval+1) * interval
	
	# Price
	c_vol = vol_dev / np.max(np.abs(vol_dev))
	plt.subplot(5,1,1)
	plt.title('Price')
	plt.scatter(np.arange(num_dates), price_list, c=c_vol, cmap='coolwarm', s=20)
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# 20MA
	plt.subplot(5,1,2)
	plt.plot(ma_list, '-o', color='tab:orange', markersize=5)
	plt.title('20MA')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# 20MA-Derivative
	plt.subplot(5,1,3)
	plt.plot(ma_dev, '-o', color='tab:red', markersize=5)
	plt.title('20MA Derivative')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# 20Vol
	plt.subplot(5,1,4)
	plt.plot(vol_list, '-o', color='tab:green', markersize=5)
	plt.title('20Vol')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# Volume
	plt.subplot(5,1,5)
	plt.plot(vol_dev, '-o', color='tab:orange', markersize=5)
	plt.title('20Vol Derivative')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	plt.tight_layout()

	jpg_path = './figures/'+code+'vcp_test2.png'
	plt.savefig(jpg_path)


def plot(
		code,
		price_list, 
		ma_list,
		vol_list, 
	):
	vol_dev = gradient(vol_list)
	ma_dev = gradient(ma_list)
	
	fig = plt.figure(figsize=(30,25))
	matplotlib.rcParams.update({'font.size': 22})

	num_dates = price_list.shape[0]
	interval = 10
	n_interval = num_dates // interval
	xticks = np.arange(0, n_interval+1) * interval
	
	# Price
	plt.subplot(5,1,1)
	plt.title('Price')
	plt.plot(price_list, '-o', color='tab:blue', markersize=4)
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# 20MA
	plt.subplot(5,1,2)
	plt.plot(ma_list, '-o', color='tab:orange', markersize=5)
	plt.title('20MA')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# 20MA-Derivative
	plt.subplot(5,1,3)
	plt.plot(ma_dev, '-o', color='tab:red', markersize=5)
	plt.title('20MA Derivative')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# 20Vol
	plt.subplot(5,1,4)
	plt.plot(vol_list, '-o', color='tab:green', markersize=5)
	plt.title('20Vol')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# Volume
	plt.subplot(5,1,5)
	plt.plot(vol_dev, '-o', color='tab:orange', markersize=5)
	plt.title('20Vol Derivative')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	plt.tight_layout()

	jpg_path = './figures/'+code+'vcp_test.png'
	plt.savefig(jpg_path)


def main():
	now = time.localtime(time.time())
	year = now.tm_year
	month = now.tm_mon
	day = now.tm_mday

	start = str(year-1) + '-' + str(month) + '-' + str(day)
	end = str(year) + '-' + str(month) + '-' + str(day)
	code_list = [
		'039030',
		'228760',
		'086520',
		'005930',
		'000660',
		'083930',
		'083310',
		'067310',
		'117730',
		'092870',
		'297090',
		'003230',
		'099190',
	]
	for code in code_list:
		print (code)
		df = get_price(code, start, end)
		price_list = list(df['Close'])[-240:]
		price_list, vol_list, ma_list = edit_(price_list)
		plot(
			code=code,
			price_list=np.asarray(price_list),
			ma_list=np.asarray(ma_list),
			vol_list=np.asarray(vol_list),
		)
		plot2(
			code=code,
			price_list=np.asarray(price_list),
			ma_list=np.asarray(ma_list),
			vol_list=np.asarray(vol_list),
		)
		exit(-1)


if __name__ == '__main__':
	main()
