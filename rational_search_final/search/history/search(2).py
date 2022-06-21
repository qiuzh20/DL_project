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
		self.url = self.baseurl.format(keywords) + '&pn={}&ie=utf-8'.format(pagen * 10)

		r = requests.get(self.url, headers = self.headers)
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
			print(data)
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
			url = item["link"]
			html = requests.get(url)
			html.encoding = 'utf-8'
			soup = BeautifulSoup(html.content, 'html.parser')
			part = soup.select('div')

			content = ''
			for paragraph in part:
				character = str(paragraph)
				soup_ = BeautifulSoup(character, 'html.parser')
				sec = soup_.select('div')[0].text
				sec = sec.replace('\n', '')
				sec = sec.replace(' ', '')
				if len(sec) > len('不如五个字'):					
					content = content +'\n'+ sec
			item['content'] = str(content)
			result.append(item)
		return result

if __name__ == "__main__":
	Keyword = '清华'
	result=f'./crawl_{Keyword}.txt'
	spider = BaiduSpider()
	num = 100
	data = spider.get_link(Keyword, num)
	data = spider.get_content(data)
	for item in data:
		spider.save_data(item, result)

#基于摘要， 分类， 评分， 的筛选 
#标题去重
#获取内容
#问题：大量的空行与空格， 乱码