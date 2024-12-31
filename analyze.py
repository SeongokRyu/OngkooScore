import time

import functools
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

import multiprocessing as mp
import parmap

from libs.RobustSTL import RobustSTL


def gradient(price_data, edge_order=2):
	first_derivative = np.gradient(price_data, edge_order=edge_order)
	return first_derivative


def get_price(
		code,
		start,
		end
	):
	df = fdr.DataReader(code, start, end)
	return df


def stl_analysis(df_price, idx_start, idx_end):
	idx_list = list(range(idx_start, idx_end))
	df_ = df_price.iloc[idx_list]

	p_close = np.asarray(df_['Close'])

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
	return derivative[-1], noise[-1]


def main():
	total_st = time.time()
	num_cores = mp.cpu_count()

	df_kospi = pd.read_csv('./raw_data/kodex200_price.csv')
	#df_kospi = df_kospi.iloc[list(range(1))]
	#df_kospi = df_kospi.tail(30)
	print (df_kospi)

	date_list = list(df_kospi['Date'])
	date_init = date_list[0]

	df = pd.read_csv('./raw_data/stock_refined_set.csv')
	condition = (df['Market'] == 'KOSPI')
	df = df[condition]
	code_list = list(df['Symbol'])
	name_list = list(df['Name'])

	df_price_list = []
	idx_list = []
	st = time.time()
	for i, code in enumerate(code_list):
		code = str(code).rjust(6, '0')
		name = name_list[i]
		df_price = pd.read_csv('./raw_data/'+code+'_price.csv')
		date_list = list(df_price['Date'])
		idx = 0
		for i, date in enumerate(date_list):
			if date == date_init:
				idx = i
				break

		if len(date_list) == 5356:
			df_price_list.append(df_price)
			idx_list.append(idx)
			print (code, name)

	et = time.time()
	print ("Time for preparing price data for STL:", round(et-st, 2))
	print ("Number of stocks to analyze STL:", len(df_price_list))

	date_list = list(df_kospi['Date'])
	print ("Number of dates to analyze:", len(date_list))

	idx_init = idx_list[0]
	trend_pos_list = []
	noise_pos_list = []
	for i, date in enumerate(date_list):
		idx_end = idx_init + i + 1
		idx_start = idx_end - 240
	
		st = time.time()
		fn_ = functools.partial(
			stl_analysis,
			idx_start=idx_start,
			idx_end=idx_end,
		)

		'''
		results = parmap.map(
			fn_,
			df_price_list,
			pm_pbar=False,
			pm_processes=num_cores,
		)
		'''
		trend_list = []
		noise_list = []
		for j, df_price in enumerate(df_price_list):
			try:
				trend, noise = stl_analysis(
					df_price=df_price,
					idx_start=idx_start,
					idx_end=idx_end
				)
				trend_list.append(trend)
				noise_list.append(noise)
			except:
				pass
		et = time.time()

		num_total = len(noise_list)
		num_noise_pos = np.sum(np.array(noise_list) > 0)
		noise_pos = float(num_noise_pos) / num_total * 100.0
		noise_pos_list.append(noise_pos)

		num_total = len(trend_list)
		num_trend_pos = np.sum(np.array(trend_list) > 0.0)
		trend_pos = float(num_trend_pos) / num_total * 100.0
		trend_pos_list.append(trend_pos)

		print (date, "\t", round(noise_pos, 3), "\t", round(trend_pos, 3), "\t", i, "/", len(date_list), round(et-st, 3))

	df_kospi['T_noise'] = noise_pos_list
	df_kospi['T_trend'] = trend_pos_list
	print (df_kospi)
	df_kospi.to_csv('kodex200_temperature.csv', index=False)
	total_et = time.time()
	print ("Time for running total executions:", round(total_et - total_st, 2), "(s)")


if __name__ == '__main__':
	main()
