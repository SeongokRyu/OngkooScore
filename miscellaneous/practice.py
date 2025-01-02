import os
import time

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from libs.RobustSTL import RobustSTL


def get_stock_list(df_path):
	if os.path.exists(df_path):
		return pd.read_csv(df_path)
	else:
		df_kospi = fdr.StockListing('KOSPI')
		df_kosdaq = fdr.StockListing('KOSDAQ')
		df_all = pd.concat([df_kospi, df_kosdaq])
		df_all.to_csv(df_path, index=False)
		return df_all


def filter_list(df):
	drop_list = []
	for i in range(len(df)):
		row = df.iloc[i]
		symbol = row['Symbol'].strip()
		if len(symbol) != 6:
			drop_list.append(i)
			continue

		if symbol[-1] != '0':
			drop_list.append(i)
			continue

		region = row['Region']
		if type(region)!=str:
			drop_list.append(i)
			continue

	df = df.drop(
		index=drop_list, 
		axis=0
	)
	print (df)

	df.to_csv('stock_clean.csv', index=False)
	return


def load_df():
	path_all = 'stock_all.csv'
	path_clean = 'stock_clean.csv'

	if os.path.exists(path_clean):
		df = pd.read_csv(path_clean)
		return df
	
	else:
		if os.path.exists(path_all):
			df = pd.read_csv(df_path)
			df = filter_list(df)
			return df
		else:
			df = get_stock_list(path_all)
			df = pd.read_csv(df_path)
			df = filter_list(df)
			return df


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


def moving_average(x, w=5):
	return np.convolve(x, np.ones(w), 'valid') / w


def stl_analysis(df_price):
	p_close = np.asarray(df_price['Close'])
	volume = np.asarray(df_price['Volume'])

	# STL for closed price
	scale = (np.max(p_close) + np.min(p_close)) / 2.0
	p_close = p_close / scale
	result = RobustSTL(
		p_close,
		season_len=1,
		reg1=10.0,
		reg2=0.5,
		K=2,
		H=5,
		dn1=1.0,
		ds1=50.0,
		ds2=1.0
	)

	trend = result[2] + 1.0
	noise = result[3]
	derivative = gradient(trend, 1)
	return p_close, trend, noise, derivative, volume


def plot(
		close,
		trend,
		noise,
		derivative,
		volume,
		code,
		name,
	):
	fig = plt.figure(figsize=(30,25))

	num_dates = close.shape[0]
	interval = 10
	n_interval = num_dates // interval
	xticks = np.arange(0, n_interval+1) * interval
	
	# Price
	plt.subplot(5,1,1)
	plt.title(code + ', ' + name + ', Scaled price')
	plt.plot(close, '-o', color='tab:blue', markersize=4)
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# Trend
	plt.subplot(5,1,2)
	plt.plot(trend, '-o', color='tab:orange', markersize=5)
	plt.title('Trend')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# Derivative
	plt.subplot(5,1,3)
	plt.plot(derivative, '-o', color='tab:red', markersize=5)
	plt.title('First derivative of Trend')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# Noise
	plt.subplot(5,1,4)
	_x = np.arange(num_dates)
	_y = np.zeros(num_dates)
	plt.plot(noise, '-o', color='tab:green', markersize=5)
	plt.plot(_x, _y, color='black')
	plt.title('Remainder (white noise)')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)
	plt.grid(True, linestyle='dashed', linewidth=0.5)

	# Volume
	plt.subplot(5,1,5)
	_x = np.arange(num_dates)
	ma = moving_average(volume, 5)
	plt.bar(_x, volume, color='tab:blue', alpha=0.5)
	plt.plot(_x[4:], ma, color='tab:orange')
	plt.title('Trading volume')
	plt.xticks(xticks)
	plt.xlim(0, num_dates)

	plt.savefig(code+'_'+name+'.png')
	return


def main():
	df = load_df()
	num_stocks = len(df)

	now = time.localtime(time.time())
	year = now.tm_year
	month = now.tm_mon
	day = now.tm_mday

	start = str(year-1) + '-' + str(month) + '-' + str(day)
	end = str(year) + '-' + str(month) + '-' + str(day)
	
	code_list = list(df['Symbol'])
	name_list = list(df['Name'])
	for i, code in enumerate(code_list):
		code = str(code).rjust(6, '0')
		name = name_list[i]
		df_price = get_price(code, start, end)
		close, trend, noise, derivative, volume = stl_analysis(df_price)

		plot(
			close=close,
			trend=trend,
			noise=noise,
			derivative=derivative,
			volume=volume,
			code=code,
			name=name,
		)
	
	return


if __name__ == '__main__':
	main()
