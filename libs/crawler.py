import csv
import sys

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


def get_code_list():
    code_list = []
    return code_list


def get_data_by_crawling(code, num_dates):
    address = "https://fchart.stock.naver.com/sise.nhn?symbol="+code+"&timeframe=day&count="+str(num_dates)+"&requestType=0"
    req = Request(address)
    res = urlopen(req)
    html = res.read().decode('cp949')
    soup = BeautifulSoup(html, "html.parser")

    items = soup.findAll('item')

    with open('/Users/seongokryu/works/finantrics/TrendAnalisys/daily_price/'+code+'.csv', mode='w') as csv_file:
        for i in items:
            data = i.get('data').split('|')
            data.append('0.0')
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(data)
    return            
