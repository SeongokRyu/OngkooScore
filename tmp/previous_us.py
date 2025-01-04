import os
import glob
import sys
import time
import warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

import img2pdf
from prettytable import PrettyTable

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib import rc

from libs.RobustSTL import RobustSTL

from functools import partial
import ray
import parmap
import multiprocessing as mp


def gradient(price_data, edge_order=2):
	first_derivative = np.gradient(price_data, edge_order=edge_order)
	return first_derivative


def stl_analysis(df_price):
	p_close = np.asarray(df_price['Close'])
	volume = np.asarray(df_price['Volume'])

	# STL for closed price
	#scale = (np.max(p_close) + np.min(p_close)) / 2.0
	scale = p_close[-1]
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


def run_stl(
		symbol,
		start,
		end,
	):
	df_price = pd.read_csv('./raw_data/'+symbol+'_price.csv')[start:end]

	close, trend, noise, derivative, volume = stl_analysis(df_price)
	noise_ = round(noise[-1]*100.0, 2)
	derivative_ = round(derivative[-1]*100.0, 2)
	return_ = round((close[-1] - close[-2]) / close[-2]*100.0, 2)
	return noise_, derivative_, return_


@ray.remote
def stl_individual_stock(
		symbol,
	):
	start_list = list(range(0,263))
	results_list = []
	for start in start_list:
		end = start+240
		try:
			results = run_stl(
				symbol=symbol,
				start=start,
				end=end,
			)
			results_list.append(results)
		except:
			return -1

	columns = [
		'noise',
		'derivative',
		'return',
	]
	df_ = pd.DataFrame(results_list, columns=columns)
	df_.to_csv('./raw_data/'+symbol+'_stl.csv', index=False)
	print (symbol, "finished")
	return 0


def main():
	price_list = list(glob.glob('./raw_data/*_price.csv'))
	symbol_list = [price.split('/')[-1].split('_')[0] for price in price_list]

	symbol_total = []
	for symbol in symbol_list:
		df_price = pd.read_csv('./raw_data/'+symbol+'_price.csv')
		if len(df_price) == 502:
			symbol_total.append(symbol)

	st = time.time()
	ray.init()
	futures = [stl_individual_stock.remote(symbol) for symbol in symbol_total]
	results = ray.get(futures)
	et = time.time()
	print (round(et-st, 2), "(s)")


if __name__ == '__main__':
	main()
