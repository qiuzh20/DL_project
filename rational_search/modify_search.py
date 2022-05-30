from search.search import BaiduSpider
from LAC import LAC
import time
import torch
from Ernie.ernie_predictor import ErniePredictor

import paddle
import argparse
import jionlp as jio

def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--keywords",
        type=str,
        required=True,
        help="Search keywords", )

    parser.add_argument(
        "--keywords_path",
        default=None,
        type=str,
        help="The path of filter keywords vocb, ./keywords_voc.txt", )
    
    parser.add_argument(
        "--Grammer_path",
        default=None,
        type=str,
        help="The path of filter keywords vocb, './grammer_voc.txt'", )

    # paramters for using classification creteria
    parser.add_argument(
        "--class_filter",
        default=False,
        type=bool,
        help="whether use classification filter", )

    parser.add_argument(
        "--task_name",
        default='seq_cls',
        type=str,
        help="The name of the task to perform predict, selected in: seq_cls and token_cls"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.", )

    parser.add_argument(
        "--model_path",
        default='./tnews_pruned_infer_model/float32',
        type=str,
        help="The path prefix of inference model to be used.", )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--use_quantize",
        action='store_true',
        help="Whether to use quantization for acceleration.", )
    parser.add_argument(
        "--set_dynamic_shape",
        action='store_true',
        help="Whether to automatically set dynamic shape.", )
    parser.add_argument(
        "--shape_info_file",
        default="shape_info.txt",
        type=str,
        help="The collected dynamic shape info file.", )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help="Whether to use fp16 inference.", )
    args = parser.parse_args()
    return args

class Vocab:
    def __init__(self, vocab_path):
        self.itos = set()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for word in f:
                w = word.strip('\n')
                # self.stoi[w] = i
                self.itos.add(w)
        print("Using {} elements filter".format(len(self.itos)))
    
    def intersection(self, inputs):
        intersection = self.itos.intersection(set(inputs))
        return intersection


class Gracab:
    def __init__(self, Grammer_path, key_score=2, struc_score=1):
        file = open(Grammer_path, 'r+')
        dic = eval(file.read())
        self.ner = set(dic["specific"])
        self.nvo = set(dic['simple'])
        self.key_scoure = key_score
        self.struc_score = struc_score
        print("Using {} elements filter".format(len(self.ner)+ len(self.nvo)))
    
    def structure_score(self, inputs):
        intersection1 = self.ner.intersection(inputs)
        len_1 = len(intersection1)
        intersection2 = self.nvo.intersection(inputs)
        len_2 = len(intersection2)
        # print("Find {} keywords, {} key structures".format(len_1, len_2))
        return len_1*self.key_scoure + len_2*self.struc_score

class Search_title_process:
    def __init__(self, args):
        self.pre_process = []
        if args.keywords_path != None:
            self.seg = LAC(mode='seg')
            self.vocab = Vocab(args.keywords_path)
            self.pre_process.append('keywords')
        if args.Grammer_path != None:
            self.lac = LAC(mode='lac')
            self.gracab = Gracab(args.Grammer_path)
            self.pre_process.append('grammer')
        if args.class_filter:
            self.predictor = ErniePredictor(args)
            self.pre_process.append('class')

    def keywords_based_filter(self, inputs):
        # filter some keywords 
        seg_result = self.seg.run(inputs)
        inters = []
        for element in seg_result:
            inter = self.vocab.intersection(element)
            inters.append(inter)
        return inters

    def structure_based_filter(self, inputs):
        ana_result = self.lac.run(inputs)
        structures_score = []
        # TODO based on learning methods, or some rule based, can consider later
        # for element in ana_result:
        def list_aug(lists, lengths=[3, 4, 5]):
            initial_length = len(lists)
            out_list = lists[:]
            for length in lengths:
                for i in range(0, initial_length-length):
                    temp = ''
                    for j in range(length):
                        temp += out_list[i+j]
                    out_list.append(temp)
            return out_list
        for element in ana_result:
            aug_list = list_aug(element[1])
            structures_score.append(self.gracab.structure_score(aug_list))
        return structures_score

    def cls_filter(self, inputs):
        infer_result = self.predictor.predict(inputs)
        classes = []
        confidences = []
        label_list = [
        "news_story", "news_culture", "news_entertainment", "news_sports",
        "news_finance", "news_house", "news_car", "news_edu", "news_tech",
        "news_military", "news_travel", "news_world", "news_stock",
        "news_agriculture", "news_game"]
        label = infer_result["label"].squeeze().tolist()
        confidence = infer_result["confidence"].squeeze().tolist()
        for i, ret in enumerate(infer_result["label"]):
            classes.append(label_list[label[i]])
            confidences.append(confidence[i])
        return classes, confidences

    def pre_processing(self, inputs):
        feature = {}
        for key in self.pre_process:
            if key == 'keywords':
                feature[key] = self.keywords_based_filter(inputs)
            elif key == 'grammer':
                feature[key] = self.structure_based_filter(inputs)
            elif key == 'class':
                feature[key], feature['confidence'] = self.cls_filter(inputs)
        return feature

    def dir_output(self, inputs):
        # print(inputs)
        feature = self.pre_processing(inputs)
        for i, element in enumerate(inputs):
            element_feature = []
            for key in feature.keys():
                element_feature.append(key + ":{}".format(feature[key][i]))
            print(element, element_feature)

class Search_context_process:
    def __init__(self, args):
        self.keywards = jio.keyphrase.extract_keyphrase
        self.pre_process = []
    def get_keywords(self, inputs):
        keyphrases = self.keywards(inputs, with_weight=True)
        return keyphrases

if __name__ == "__main__":
    args = parse_args()
    args.device = 'gpu'
    spider = BaiduSpider()
    spider.search(args.keywords)
    Search_process = Search_title_process(args)
    begin = time.time()
    Search_process.dir_output(spider.titles)
    finish_search = time.time()
    print("finish in {}".format(finish_search - begin))