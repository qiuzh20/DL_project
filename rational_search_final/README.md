# 理性搜索：NLP技术的综合运用

## 运行指令

在交互式界面中显示（推荐）

```Shell
CUDA_VISIBLE_DEVICES=[GPU index] streamlit web_app_v2.py

[等待反馈]
  You can now view your Streamlit app in your browser.

  Network URL: http://192.168.6.111:8501 (点击前往对应浏览器界面操作)
  External URL: http://166.111.80.138:8501
```

在命令行中操作

```Shell
CUDA_VISIBLE_DEVICES=[GPU index] python modify_search_v2.py
(等待模型载入)
请输入搜索关键词+回车，直接按回车结束：
[输入关键词]
请输入输出类型，'emo'+回车 按情感分类输出
[输入输出需求，目前支持全输出与按照情感倾向分类输出]
使用 {}s 完成处理，若需要继续，请输入搜索关键词+回车，直接按回车结束
......
```

## 环境配置

```shell
conda env create -f rational_search.yaml
```

## References

- 情感分类，新闻分类：[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP), [Ernie3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)
- 分词与词性分析：[百度lac](https://github.com/baidu/lac), [pkuseg](https://github.com/lancopku/PKUSeg-python)
- 摘要生成：[基于bert的抽取式摘要生成](https://github.com/jasoncao11/nlp-notebook/tree/master/4-6.Bert-extractive-summarizer), [基于GPT2的生成式摘要生成](https://github.com/qingkongzhiqian/GPT2-Summary)
- 关键词摘取：[CKPE](https://github.com/dongrixinyu/chinese_keyphrase_extractor)
