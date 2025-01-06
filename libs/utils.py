import os
import argparse
import numpy as np
import FinanceDataReader as fdr

import img2pdf
from prettytable import PrettyTable


def str2bool(v):
	if v.lower() in ['yes', 'true', 't', 'y', '1']:
		return True
	elif v.lower() in ['no', 'false', 'f', 'n', '0']:
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected')


def print_pretty_table(df):
	table = PrettyTable([''] + list(df.columns))
	for row in df.itertuples():
		table.add_row(list(row))
	print (str(table))


def combine_to_pdf(
		jpg_list, 
		pdf_path
	):
	with open(pdf_path, 'wb') as f:
		f.write(img2pdf.convert(jpg_list))


def get_price(
		code,
		start,
		end
	):
	df = fdr.DataReader(code, start, end)
	return df


def gradient(
		price_data, 
		edge_order=2
	):
	first_derivative = np.gradient(price_data, edge_order=edge_order)
	return first_derivative


def moving_average(x, w=5):
	return np.convolve(x, np.ones(w), 'valid') / w
