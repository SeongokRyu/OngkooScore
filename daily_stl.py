import os
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
#from matplotlib import font_manager
#font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#font_fname = font_list[106]
#font_name = font_manager.FontProperties(fname=font_fname).get_name()
#rc('font', family=font_name)

from libs.RobustSTL import RobustSTL

#UPDATE_KOSPI = False
UPDATE_KOSPI = True

def print_pretty_table(df):
	table = PrettyTable([''] + list(df.columns))
	for row in df.itertuples():
		table.add_row(list(row))
	print (str(table))


def combine_to_pdf(jpg_list, pdf_path):
	with open(pdf_path, 'wb') as f:
		f.write(img2pdf.convert(jpg_list))


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


def update_kospi_temperature(
		noise,
		trend,
		mix,
		start,
		end,
	):
	df_total = pd.read_csv('./results/kospi_with_temperature.csv')
	df_kospi = fdr.DataReader('069500', start, end)

	date_list = list(df_total['Date'])
	close_list = list(df_total['Close'])
	noise_list = list(df_total['T_noise'])
	trend_list = list(df_total['T_trend'])
	mix_list = list(df_total['T_mix'])

	date_list.append(end)
	close_list.append(df_kospi.iloc[-1]['Close'])
	noise_list.append(noise)
	trend_list.append(trend)
	mix_list.append(mix)

	df_update = pd.DataFrame({})
	df_update['Date'] = date_list
	df_update['Close'] = close_list
	df_update['T_noise'] = noise_list
	df_update['T_trend'] = trend_list
	df_update['T_mix'] = mix_list

	df_update.to_csv('./results/kospi_with_temperature.csv', index=False)
	print (df_update)


def summarize(
		return_list,
		noise_list,
		derivative_list,
		start,
		end,
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
	if UPDATE_KOSPI:
		update_kospi_temperature(
			noise=noise_,
			trend=trend_,
			mix=mix_,
			start=start,
			end=end,
		)


def main(args):
	st = time.time()
	csv_path = args.input_csv
	df = pd.read_csv(csv_path)
	code_list = list(df['Symbol'])
	name_list = list(df['Name'])

	start = str(args.year-1) + '-' + str(args.month) + '-' + str(args.day)
	end = str(args.year) + '-' + str(args.month) + '-' + str(args.day)

	print (start, " ~ ", end)
	
	noise_list = []
	derivative_list = []
	return_list = []
	code_list_re = []
	name_list_re = []
	jpg_list = []
	for i, code in enumerate(code_list):
		code = str(code).rjust(6, '0')
		name = name_list[i]
		df_price = get_price(code, start, end)
		try:
			close, trend, noise, derivative, volume = stl_analysis(df_price)

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

			noise_list.append(noise_)
			return_list.append(return_)
			derivative_list.append(derivative_)
			code_list_re.append(code)
			name_list_re.append(name)
			jpg_list.append(jpg_path)

			print (code, '\t', name, "\t Noise:", noise_, "\t Derivative:", derivative_, "\t Return:", return_, "\t",  i+1, "/", len(code_list))
		except:
			print (code, '\t', name, " terminated with error")
			pass

	df_final = pd.DataFrame({})
	df_final['Code'] = code_list_re
	df_final['Name'] = name_list_re
	df_final['Return'] = return_list
	df_final['Noise'] = noise_list
	df_final['Derivative'] = derivative_list
	df_final = df_final.sort_values(by='Noise', ascending=True)
	csv_path  = './results/STL_'+end+'.csv'
	df_final.to_csv(csv_path, index=False)

	print_pretty_table(df_final)

	pdf_path  = './results/STL_'+end+'.pdf'
	combine_to_pdf(
		jpg_list=jpg_list,
		pdf_path=pdf_path
	)

	summarize(
		return_list=return_list,
		noise_list=noise_list,
		derivative_list=derivative_list,
		start=start,
		end=end,
	)
	et = time.time()
	print ("Time for running:", round(et-st, 2), " (s)")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_csv', type=str, required=True, 
						help='')
	parser.add_argument('-y', '--year', type=int, required=True,
						help='')
	parser.add_argument('-m', '--month', type=int, required=True,
						help='')
	parser.add_argument('-d', '--day', type=int, required=True,
						help='')
	args = parser.parse_args()

	main(args)
