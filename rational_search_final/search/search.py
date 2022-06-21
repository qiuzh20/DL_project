# coding=utf-8
import sys
from bs4 import BeautifulSoup
import requests
from lxml import etree
import re
import urllib
import xlwt
import time

class BaiduSpider(object):
	def __init__(self):
		self.key = None
		self.baseurl ='https://www.baidu.com/s?rtt=1&bsst=1&cl=2&tn=news&ie=utf-8&word={}'
		self.url = self.baseurl
		self.headers = {
			'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"
		}

	def get_html(self, keywords, pagen):
		self.key = keywords
		self.url = self.baseurl.format(keywords) + '&pn={}&ie=utf-8'.format(pagen)

		r = requests.get(self.url, headers = self.headers, verify=False)
		r.encoding = 'utf-8'
		res=etree.HTML(r.text)
		selector=res.xpath('//div[@id="content_left"]/div[@class="result-op c-container xpath-log new-pmd"]')
		datalist = []

		for data in selector:
			item = {}
			item["title"] = ''.join(data.xpath('./div/h3/a/@aria-label'))
			item['link']=''.join(data.xpath('./div/h3/a/@href'))
			datalist.append(item)
		return datalist

	def save_data(self,item,filename):
		with open(filename,'a',encoding='utf-8')as f:
			data=item['title'] + '\n' + item['link'] + '\n' + item['content'] + '\n'
			#print(data)
			f.write(data+'\n')

	def get_link(self, Keywords, num):
		n = num // 10
		data = []
		for i in range(n):
			pagen = i * 10
			datalist = self.get_html(Keywords, pagen)
			for item in datalist:
				data.append(item)
		data = self.unique(data)
		return data

	def countContent(self, para):
		pattern = re.compile(u'[\u1100-\uFFFD]+?')
		content = pattern.findall(para)
		return content

	def unique(self, data):
		result = []
		result.append(data[0])
		for item in data:
			check = True
			for i in result:
				if item['title'] == i['title']:
					check = False
			if check:
				result.append(item)
		return result

	def get_content(self, data):
		result = []
		for item in data:
			url = item['link']
			html = requests.get(url, headers= self.headers, verify=False)
			html.encoding = 'utf-8'
			tree = BeautifulSoup(html.content, 'lxml')

			
			body = tree.body
			if body is None:
				continue

			for tag in body.select('script'):
				tag.decompose()
			for tag in body.select('style'):
				tag.decompose() 
			text = body.get_text(separator='\n')
			list = text.split()
			sample = '字' * 50
			text = ''
			for i in list:
				if len(i) >= len(sample):
					text = text + i
			if text != '':
				item['content'] = text
				result.append(item)
		return result

class BingSpider(object):
	def __init__(self):
		self.key = None
		self.baseurl ='https://cn.bing.com/search?q={}'
		self.url = self.baseurl
		self.headers = {
			'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"
		}

	def get_html(self, keywords, pagen):
		self.key = keywords
		self.url = self.baseurl.format(keywords) + '&&first={}&FORM=PORE'.format(pagen)

		r = requests.get(self.url, headers = self.headers, verify=False)
		r.encoding = 'utf-8'
		tree=etree.HTML(r.text)
		list = tree.xpath('//ol[@id="b_results"]//li[@class="b_algo"]//div[@class= "b_title"]')

		data = []
		for title in list:
			item = {}
			h3 = title.xpath('./h2/a')[0]
			h3 = h3.xpath('string(.)')
			link = ''.join(title.xpath('./h2/a/@href'))
			item['title'] = h3
			item['link'] = link
			data.append(item)

		return data

	def save_data(self,item,filename):
		with open(filename,'a',encoding='utf-8')as f:
			data=item['title'] + '\n' + item['link'] + '\n' + item['content'] + '\n'
			#print(data)
			f.write(data+'\n')

	def get_link(self, Keywords, num):
		n = num // 10
		data = []
		for i in range(n):
			pagen = i * 10 + 9
			datalist = self.get_html(Keywords, pagen)
			for item in datalist:
				data.append(item)
		data = self.unique(data)
		return data

	def countContent(self, para):
		pattern = re.compile(u'[\u1100-\uFFFD]+?')
		content = pattern.findall(para)
		return content

	def unique(self, data):
		result = []
		result.append(data[0])
		for item in data:
			check = True
			for i in result:
				if item['title'] == i['title']:
					check = False
			if check:
				result.append(item)
		return result

	def get_content(self, data):
		result = []
		for item in data:
			url = item['link']
			html = requests.get(url, headers= self.headers, verify=False)
			html.encoding = 'utf-8'
			tree = BeautifulSoup(html.content, 'lxml')

			
			body = tree.body
			if body is None:
				continue

			for tag in body.select('script'):
				tag.decompose()
			for tag in body.select('style'):
				tag.decompose() 
			text = body.get_text(separator='\n')
			list = text.split()
			sample = '字' * 50
			text = ''
			for i in list:
				if len(i) >= len(sample):
					text = text + i
			if text != '':
				item['content'] = text
				result.append(item)
		return result


if __name__ == "__main__":
	Keyword = '上海'
	result=f'./crawl_{Keyword}.txt'
	spider = BingSpider()
	num = 200
	data = spider.get_link(Keyword, num)
	data = spider.get_content(data)
	for item in data:
		spider.save_data(item,result)
