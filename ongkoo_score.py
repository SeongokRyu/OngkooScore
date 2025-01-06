import os
import sys
import time
import warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

import ray
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import rc

from libs.RobustSTL import RobustSTL
from libs.utils import str2bool
from libs.utils import print_pretty_table
from libs.utils import combine_to_pdf
from libs.utils import get_price
from libs.utils import gradient
from libs.utils import moving_average


INPUT_DICT = {
	'kr': os.path.join(os.getcwd(), 'raw_data', 'interest.csv'),
	'us': os.path.join(os.getcwd(), 'raw_data', 'snp500.csv'),
}

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


def plot(
		close,
		trend,
		noise,
		derivative,
		volume,
		code,
		name,
		jpg_path,
	):
	fig = plt.figure(figsize=(30,25))
	matplotlib.rcParams.update({'font.size': 22})

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
	plt.tight_layout()

	plt.savefig(jpg_path)
	return 


def update(
		market,
		noise,
		trend,
		mix,
		start,
		end,
	):
	df_total = pd.read_csv('./results/'+market+'_with_temperature.csv')
	df_market = None
	if market == 'kr':
		df_market = fdr.DataReader('069500', start, end)
	elif market == 'us':
		df_market = fdr.DataReader('S&P500', start, end)

	date_list = list(df_total['Date'])
	close_list = list(df_total['Close'])
	noise_list = list(df_total['T_noise'])
	trend_list = list(df_total['T_trend'])
	mix_list = list(df_total['T_mix'])

	date_list.append(end)
	close_list.append(df_market.iloc[-1]['Close'])
	noise_list.append(noise)
	trend_list.append(trend)
	mix_list.append(mix)

	df_update = pd.DataFrame({})
	df_update['Date'] = date_list
	df_update['Close'] = close_list
	df_update['T_noise'] = noise_list
	df_update['T_trend'] = trend_list
	df_update['T_mix'] = mix_list

	df_update.to_csv('./results/'+market+'_with_temperature.csv', index=False)
	print (df_update)


def summarize(
		market,
		return_list,
		noise_list,
		derivative_list,
		start,
		end,
		update_latest,
	):
	num_return_pos = np.sum(np.array(return_list) > 0)
	num_return_neg = len(return_list) - num_return_pos
	temperature_return = round(100.0* num_return_pos / len(return_list), 1)

	num_noise_pos = np.sum(np.array(noise_list) > 0)
	num_noise_neg = len(noise_list) - num_noise_pos
	temperature_noise = round(100.0* num_noise_pos / len(noise_list), 1)

	num_derivative_pos = np.sum(np.array(derivative_list) > 0)
	num_derivative_neg = len(derivative_list) - num_derivative_pos
	temperature_derivative = round(100.0* num_derivative_pos / len(derivative_list), 1)
	
	print ("Summarize:")
	print ("Noise,      (+):", num_noise_pos, " (-):", num_noise_neg, " Temperature", temperature_noise)
	print ("Derivative, (+):", num_derivative_pos, " (-):", num_derivative_neg, " Temperature", temperature_derivative)
	print ("Return,     (+):", num_return_pos, " (-):", num_return_neg, " Temperature", temperature_return)

	noise_ = temperature_noise*0.01
	trend_ = temperature_derivative*0.01
	mix_ = (noise_ + trend_) * 0.5
	if update_latest:
		update(
			market=market,
			noise=noise_,
			trend=trend_,
			mix=mix_,
			start=start,
			end=end,
		)

@ray.remote
def run_stl(
		inp,
		start,
		end,
	):
	code, name = inp[0], inp[1]
	try:
		path = os.path.join(
			os.getcwd(),
			'tmp_data',
			code+'_price.csv'
		)
		df_price = pd.read_csv(path)
		close, trend, noise, derivative, volume = stl_analysis(df_price=df_price)

		jpg_path = './figures/'+code+'_'+name+'.jpg'
		plot(
			close=close,
			trend=trend,
			noise=noise,
			derivative=derivative,
			volume=volume,
			code=code,
			name=name,
			jpg_path=jpg_path,
		)
	
		noise_ = round(noise[-1]*100.0, 2)
		derivative_ = round(derivative[-1]*100.0, 2)
		return_ = round((close[-1] - close[-2]) / close[-2]*100.0, 2)
		print (code, '\t', name, "\t Noise:", noise_, "\t Derivative:", derivative_, "\t Return:", return_)
		return True, noise_, derivative_, return_
	except:
		print (code, '\t', name, " terminated with error")
		return False, -1, -1, -1


def main(args):
	st_all = time.time()

	df = pd.read_csv(INPUT_DICT[args.market])
	code_list = list(df['Symbol'])
	name_list = list(df['Name'])

	year, month, day = args.date.split('-')
	start = str(int(year)-1) + '-' + month + '-' + day
	end = args.date
	print (start, " ~ ", end)

	st = time.time()
	code_list_re = []
	for code in code_list:
		if args.market == 'kr':
			code = str(code).rjust(6, '0')
		code_list_re.append(code)
		df_price = get_price(code=code, start=start, end=end)
		tmp_path = os.path.join(
			os.getcwd(),
			'tmp_data',
			code+'_price.csv'
		)
		df_price.to_csv(tmp_path, index=False)
	et = time.time()
	print ("Finish download price data, Time spent:", round(et-st, 2), "(s)")
	
	st = time.time()
	ray.init()
	inp_list = list(zip(code_list_re, name_list))
	futures = [run_stl.remote(inp, start, end) for inp in inp_list]
	results = ray.get(futures)
	ray.shutdown()
	et = time.time()
	print ("Finish running STL, Time spent:", round(et-st, 2), "(s)")

	code_final = []
	name_final = []
	noise_final = []
	derivative_final = []
	return_final = []
	for idx, result in enumerate(results):
		terminate_, noise_, derivative_, return_ = result[0], result[1], result[2], result[3]
		if terminate_:
			code_final.append(code_list_re[idx])
			name_final.append(name_list[idx])
			noise_final.append(noise_)
			derivative_final.append(derivative_)
			return_final.append(return_)

	df_final = pd.DataFrame({})
	df_final['Code'] = code_final
	df_final['Name'] = name_final
	df_final['Return'] = return_final
	df_final['Noise'] = noise_final
	df_final['Derivative'] = derivative_final
	df_final = df_final.sort_values(by='Noise', ascending=True)

	final_path  = './results/STL_'+args.market+'_'+end+'.csv'
	df_final.to_csv(final_path, index=False)

	print_pretty_table(df_final)

	summarize(
		market=args.market,
		return_list=return_final,
		noise_list=noise_final,
		derivative_list=derivative_final,
		start=start,
		end=end,
		update_latest=args.update_latest,
	)
	et_all = time.time()
	print ("Time for running:", round(et_all-st_all, 2), " (s)")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--market', type=str, required=True,
						help='Market, Options: kr/us')
	parser.add_argument('-d', '--date', type=str, required=True,
						help='Please give the date format as 2025-01-01')
	parser.add_argument('-u', '--update_latest', type=str2bool, default=True,
						help='')
	args = parser.parse_args()

	main(args)
