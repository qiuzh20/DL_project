# 理性搜索：NLP技术的综合运用

## 运行指令

```Shell
python modify_search.py --keywords 疫情 \
--model_path ./Ernie/tnews_pruned_infer_model/float32 \
--keywords_path ./keywords_voc.txt \
--Grammer_path ./grammer_voc.txt  \
--class_filter True  > lac2.txt
```

## TODO

- `modify_search.py` 部分:
  1. 补充对于搜索内容的处理(spider.content)
  2. 完善`Search_context_process`类（调用Bert-extractive-summarizer  MultiLable-Classification 中准备好接口，可以直接对于输入为列表的多条内容进行处理）、
  最好能够支持batch处理，也可以考虑先串行处理
  3. 补充更加完善的search类，综合调用`Search_title_process`与`Search_context_process`
- `Bert-extractive-summarizer`:
  1. 做好接口
  2. 争取支持batch
- `MultiLable-Classification`:
  1. 做好接口
  2. 争取支持batch
- `Emotion_cls`:
  1. 做好接口
  2. 争取支持batch

## Method

## Demo

## References
