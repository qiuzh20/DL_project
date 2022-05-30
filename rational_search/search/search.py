# coding=utf-8
import sys
from bs4 import BeautifulSoup
import requests
from lxml import etree
import urllib
import xlwt
import time
from LAC import LAC

class BaiduSpider(object):
	def __init__(self):
		self.key = None
		self.baseurl ='https://www.baidu.com/s?rtt=1&bsst=1&cl=2&tn=news&ie=utf-8&word={}'
		self.url = self.baseurl
		self.titles = []

	def get_html(self, keywords, pagen):
		self.key = keywords
		self.url = self.baseurl.format(keywords) + '&pn={}&ie=utf-8'.format(pagen * 10)

		headers = {
			'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"
		}
		r = requests.get(self.url, headers = headers)
		r.encoding = 'utf-8'
		res=etree.HTML(r.text)
		selector=res.xpath('//div[@id="content_left"]/div[@class="result-op c-container xpath-log new-pmd"]')
		datalist = []

		for data in selector:
			item = {}
			item["title"] = "".join(data.xpath('./div/h3/a/@aria-label'))
			item['link']= "".join(data.xpath('./div/h3/a/@href'))
			self.titles.append(item["title"][3:])
			datalist.append(item)
		return datalist

	def save_data(self,item,filename):
		with open(filename,'a',encoding='utf-8')as f:
			data=item['title'] + '\n' + item['link']
			# print(data)
			f.write(data+'\n')

	def get_link(self, Keywords, num):
		n = num // 10
		data = []
		for i in range(n):
			pagen = i * 10
			datalist = self.get_html(Keywords, pagen)
			for item in datalist:
				data.append(item)
		return data

	def get_content(self, data):
		for item in data:
			url = item["link"]

	def search(self, keyword):
		result = "./crawl_{}.txt".format(keyword)
		n=1
		while True:
			data_list=self.get_html(keyword ,n)
			# for data in data_list:
			# 	spider.save_data(data, result)
			# time.sleep()
			if n <= 3:
				n+=1
			else:
				print(f'程序已经退出，在{int(n/10)+1}页......')
				break
		# print(self.titles)
	

if __name__ == "__main__":
	Keyword = "北京疫情"
	spider = BaiduSpider()
	spider.search(Keyword)
 
	lac = LAC(mode='rank')
	rank_result = lac.run(spider.titles)
	print(rank_result)