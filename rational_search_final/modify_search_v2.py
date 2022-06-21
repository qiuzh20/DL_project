from search.search import BaiduSpider
from search.search import BingSpider
from LAC import LAC
import time
import torch

import paddle
import argparse

import logging


def parse_args():
    parser = argparse.ArgumentParser()
    ##############################################################################
    #                      Basic Searching paramerters                           #
    ##############################################################################
    parser.add_argument(
        "--num",
        default=200,
        type=int,
        help="search candidates number", )
    parser.add_argument(
        "--spider",
        default="百度",
        type=str,
        help="use search engine", )
    ##############################################################################
    #                         title processing parameters                        #
    ##############################################################################
    parser.add_argument(
        "--keywords_filter",
        action='store_false',
        help="whether to use keywords filter vocb, see ./knowledge_based/keywords_voc.txt", )
    parser.add_argument(
        "--grammer_filter",
        action='store_false',
        help="whether to use grammmer filter vocb, see ./knowledge_based/grammer_voc.txt'", )
    parser.add_argument(
        "--class_filter",
        action='store_false',
        help="whether use classification filter", )

    ##############################################################################
    #                         context processing parameters                        #
    ##############################################################################
    parser.add_argument(
        "--extract_keywords",
        action='store_false',
        help='whether to use keywords extractor for text process'
    )
    parser.add_argument(
        "--emotion_cls",
        action='store_false',
        help='whether to classify emotion'
    )
    parser.add_argument(
        "--extract_summary",
        action='store_true',
        help='whether to summary long text'
    )
    parser.add_argument(
        "--generate_summary",
        action='store_true',
        help='whether to summary long text'
    )
    parser.add_argument(
        "--cls",
        action='store_false',
        help='whether to do content classification'
    )

    ##############################################################################
    #       other default parameters (don't change unless with confidence)       #
    ##############################################################################
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
        default='./Ernie/tnews_pruned_infer_model/float32',
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
    '''key words based title analyse'''
    def __init__(self, vocab_path, key_score=-2):
        self.itos = set()
        self.key_score = key_score
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for word in f:
                w = word.strip('\n')
                # self.stoi[w] = i
                self.itos.add(w)
        print("Using {} elements filter".format(len(self.itos)))
    
    def intersection(self, inputs):
        intersection = self.itos.intersection(set(inputs))
        return len(intersection)*self.key_score

class Gracab:
    '''Grammer based title analyse, would return high score if having clear structure'''
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
    '''process titles, support three operation
        keywords_filter
        grammer_filter
        class_filter
        These filters' results can be used for down streaming works
        like customized outputs'''
    def __init__(self, args):
        self.pre_process = []
        if args.keywords_filter:
            self.seg = LAC(mode='seg')
            self.vocab = Vocab('./knowledge_based/keywords_voc.txt')
            self.pre_process.append('keywords')
        if args.grammer_filter:
            self.lac = LAC(mode='lac')
            self.gracab = Gracab('./knowledge_based/grammer_voc.txt')
            self.pre_process.append('grammer')
        if args.class_filter:
            dep = time.time()
            from Ernie.ernie_predictor import ErniePredictor
            self.predictor = ErniePredictor(args)
            self.pre_process.append('class')
            args.logger.info("Finish load title classification in {}".format(time.time()-dep))

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
        feature = self.pre_processing(inputs)
        for i, element in enumerate(inputs):
            element_feature = []
            for key in feature.keys():
                element_feature.append(key + ":{}".format(feature[key][i]))
            print(element, element_feature)

class Search_context_process:
    def __init__(self, args):
        self.process = []
        self.summary = []
        self.logger =args.logger
        if args.extract_keywords:
            dep = time.time()
            # import jionlp as jio
            # self.keywards = jio.keyphrase.extract_keyphrase
            import ckpe
            self.keywards = ckpe.ckpe()
            self.process.append('extract_keywords')
            self.logger.info("Finish load keywords extractor in {}".format(time.time()-dep))
        if args.extract_summary:
            dep = time.time()
            from Bert_extractive_summarizer.model_processors import ModelProcessor
            self.content_summarizer = ModelProcessor()
            self.process.append('get_summary')
            self.logger.info("Finish load summarizer in {}".format(time.time()-dep))
        if args.generate_summary:
            dep = time.time()
            from GPT2_Summary.interact import Generate_summarizer
            self.content_summarizer = Generate_summarizer()
            self.process.append('get_summary')
            self.logger.info("Finish load summarizer in {}".format(time.time()-dep))
        if args.emotion_cls:
            if args.extract_summary or args.generate_summary:
                dep = time.time()
                from Emotion.predict import emotion_predictor
                self.emo_predictor = emotion_predictor()
                self.process.append('emotion_cls')
                self.logger.info("Finish load emtion classification in {}".format(time.time()-dep))
            else:
                self.logger.warning("Can't do emtion classification without summary")
                print("Can't do emtion classification without summary")
        if args.cls:
            if args.extract_summary or args.generate_summary:
                from Ernie.ernie_predictor import ErniePredictor
                dep = time.time()
                new_args = parse_args()
                new_args.device = 'gpu'
                new_args.max_seq_length = 256
                self.predictor = ErniePredictor(new_args)
                self.process.append('cls')
                self.logger.info("Finish load classification in {}".format(time.time()-dep))
            else:
                self.logger.warning("Can't do classification without summary")
                print("Can't do classification without summary")
        
    def get_keywords(self, inputs):
        dep = time.time()
        keyphrases = []
        for content in inputs:
            keyphrases.append(self.keywards.extract_keyphrase(content))
        self.logger.info("Finish get keywords in {}".format(time.time()-dep))
        return keyphrases
    
    def emotion(self, inputs):
        # results(lable), confidence
        dep = time.time()
        results = self.emo_predictor.emo_cls(inputs, batch_size=16)
        self.logger.info("Finish predict emotion in {}".format(time.time()-dep))
        return results

    def summarize(self, inputs, sent_num=3):
        dep = time.time()
        results = []
        for content in inputs:
            results.append(self.content_summarizer(content, num_sentences=sent_num))
        self.logger.info("Finish summarizing in {}".format(time.time()-dep))
        return results

    def summarize(self, inputs):
        dep = time.time()
        results = []
        for content in inputs:
            results.append(self.content_summarizer(content))
        self.logger.info("Finish summarizing in {}".format(time.time()-dep))
        return results
    
    def cls_filter(self, inputs):
        dep = time.time()
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
        self.logger.info("Finish classification in {}".format(time.time()-dep))
        return classes, confidences

    def pre_processing(self, inputs):
        feature = {}
        for key in self.process:
            if key == 'extract_keywords':
                feature[key] = self.get_keywords(inputs)
            elif key == 'get_summary':
                feature[key] = self.summarize(inputs)
                self.summary = feature[key]
            elif key == 'emotion_cls':
                feature[key], feature['emotion confidence'] = self.emotion(self.summary)
            elif key == 'cls':
                feature[key], feature['confidence'] = self.cls_filter(self.summary)
        return feature
    
    def dir_output(self, inputs):
        # print(inputs)
        feature = self.pre_processing(inputs)
        for i, element in enumerate(inputs):
            element_feature = []
            for key in feature.keys():
                element_feature.append(key + ":{}".format(feature[key][i]))
            print(element, element_feature)

class Rational_search:
    def __init__(self, args):
        self.logger =args.logger
        self.title_process = Search_title_process(args)
        self.content_process = Search_context_process(args)
        if args.spider == "百度":
            self.spider = BaiduSpider()
        else:
            self.spider = BingSpider()
        self.ini_titles = []
        self.ini_contents = []

    def run(self, keywards, elements_num=200, out_type='initial'):
        
        # TODO: support distributed processing to accelerate

        # basic initialization
        self.ini_titles = []
        self.ini_contents = []
        filename = './search_results/{}-{}.txt'.format(keywards, out_type)

        # get link and title
        begin_link = time.time()
        data = self.spider.get_link(keywards, elements_num)
        end_link = time.time()
        self.logger.info("Finish title search in {}".format(end_link-begin_link))
        
        # get content
        data = self.spider.get_content(data)
        print("Find {} articles".format(len(data)))
        end_content = time.time()
        self.logger.info("Finish content search in {}".format(end_content-end_link))

        # process titles
        for item in data:
            self.ini_titles.append(item['title'][3:])
        title_features = self.title_process.pre_processing(self.ini_titles)
        end_title = time.time()
        self.logger.info("Finish title processing in {}".format(end_title-end_content))

        # process contents
        for item in data:
            self.ini_contents.append(item['content'])
        content_features = self.content_process.pre_processing(self.ini_contents)
        end_feature = time.time()
        self.logger.info("Finish content processing in {}".format(end_feature-end_title))

        # output results
        # TODO: support more customized output
        if out_type == 'initial':
            return self.all_output(data, title_features, content_features, filename)
        elif out_type == 'emo':
            if 'emotion_cls' in self.content_process.process:
                return self.emo_output(data, title_features, content_features, filename)
            else:
                print("Can't do emo_output without content emotion classification! Initial output instead")

    # directly return results
    def all_output(self, data, title_features, content_features, filename):
        pairs = []
        text_out = ""
        for i, item in enumerate(data):
            title_f = {}
            content_f = {}
            
            # load title features
            for key in title_features.keys():
                title_f[key] = title_features[key][i]
            # self.logger.info(title_f)
            # load content features
            for key in content_features.keys():
                content_f[key] = content_features[key][i]
            
            pairs.append([item, title_f, content_f])

            # writing in files
            with open(filename,'a',encoding='utf-8')as f:
                writing=item['title'] + '\n' + item['link'] + '\n' 
                for ___ in title_f.keys():
                    # print(___)
                    # print(title_f[___])
                    temp = str(___) + '  ' + str(title_f[___]) + ' '
                    writing += temp
                writing += '\n'
                for __ in content_f:
                    writing += __ + '\n'
                writing += item['content'] + '\n'
                text_out += writing + '\n'
                f.write(writing+'\n')
        return pairs, text_out
    # classify based on emotional tendency
    def emo_output(self, data, title_features, content_features, filename):
        pairs = []
        pos_writings = []
        pos_pairs = []
        text_out = ""
        for i, item in enumerate(data):
            title_f = {}
            content_f = {}
            
            # to record negtive tendency contents
            neg = False
            for key in title_features.keys():
                title_f[key] = title_features[key][i]
            for key in content_features.keys():
                if key == 'emotion_cls' and content_features[key][i] == 'negative':
                    neg = True
                content_f[key] = content_features[key][i]
            if neg:
                pairs.append([item, title_f, content_f])
                with open(filename,'a',encoding='utf-8')as f:
                    writing=item['title'] + '\n' + item['link'] + '\n'
                    for ___ in title_f.keys():
                        temp = str(___) + '  ' + str(title_f[___]) + ' '
                        writing += temp
                    writing += '\n'
                    for __ in content_f:
                        writing += __ + '\n'
                    writing += item['content'] + '\n'
                    f.write(writing+'\n')
                    text_out += writing + '\n'
            # save positive content and write later
            else:
                pos_pairs.append([item, title_f, content_f])
                writing=item['title'] + '\n' + item['link'] + '\n'
                for ___ in title_f.keys():
                    temp = str(___) + '  ' + str(title_f[___]) + ' '
                    writing += temp
                writing += '\n'
                for __ in content_f:
                    writing += __ + '\n'
                writing += item['content'] + '\n'
                pos_writings.append(writing)
        for element in pos_writings:
            with open(filename,'a',encoding='utf-8')as f:
                f.write(element+'\n')
            text_out += element + '\n'
        return (pairs, pos_pairs), text_out

if __name__ == "__main__":
    args_ = parse_args()
    args_.device = 'gpu'
    logging.basicConfig(filename="./search_logs/{}.log".format(time.time()), filemode="w", 
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    args_.logger = logger
    search_engine = Rational_search(args_)
    command = input("请输入搜索关键词+回车，直接按回车结束\n")
    out_cluster = input("请输入输出类型，'emo'+回车 按情感分类输出\n")
    while len(command)>0:
        begin_time = time.time()
        if out_cluster == 'emo':
            search_engine.run(command, out_type='emo')
        else:
            search_engine.run(command)
        command = input("使用 {}s 完成处理，若需要继续，请输入搜索关键词+回车，直接按回车结束 \n".format(time.time()-begin_time))
        if len(command)>0:
            out_cluster = input("请输入输出类型，'emo'+回车 按情感分类输出\n")