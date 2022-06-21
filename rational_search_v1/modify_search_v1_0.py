from search.search import BaiduSpider
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
        "--keywords",
        required=True,
        type=str,
        help="Search keywords", )
    parser.add_argument(
        "--num",
        default=200,
        type=int,
        help="search candidates number", )
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
    #                         title processing parameters                        #
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
        "--get_summary",
        action='store_false',
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

args = parse_args()
args.device = 'gpu'

logging.basicConfig(filename="./search_logs/{}.log".format(args.keywords), filemode="w", 
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


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
            logging.info("Finish load title classification in {}".format(time.time()-dep))

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
        self.process = []
        self.summary = []
        if args.extract_keywords:
            dep = time.time()
            import jionlp as jio
            self.keywards = jio.keyphrase.extract_keyphrase
            self.process.append('extract_keywords')
            logging.info("Finish load keywords extractor in {}".format(time.time()-dep))
        if args.get_summary:
            dep = time.time()
            from Bert_extractive_summarizer.model_processors import ModelProcessor
            self.content_summarizer = ModelProcessor()
            self.process.append('get_summary')
            logging.info("Finish load summarizer in {}".format(time.time()-dep))
        if args.emotion_cls:
            if args.get_summary:
                dep = time.time()
                from Emotion.predict import emotion_predictor
                self.emo_predictor = emotion_predictor()
                self.process.append('emotion_cls')
                logging.info("Finish load emtion classification in {}".format(time.time()-dep))
            else:
                logging.warning("Can't do emtion classification without summary")
        if args.cls:
            if args.get_summary:
                from Ernie.ernie_predictor import ErniePredictor
                dep = time.time()
                new_args = parse_args()
                new_args.device = 'gpu'
                new_args.max_seq_length = 256
                self.predictor = ErniePredictor(new_args)
                self.process.append('cls')
                logging.info("Finish load classification in {}".format(time.time()-dep))
            else:
                logging.warning("Can't do classification without summary")
        
    def get_keywords(self, inputs):
        dep = time.time()
        keyphrases = []
        for content in inputs:
            keyphrases.append(self.keywards(content))
        logging.info("Finish get keywords in {}".format(time.time()-dep))
        return keyphrases
    
    def emotion(self, inputs):
        # results(lable), confidence
        dep = time.time()
        results = self.emo_predictor.emo_cls(inputs, batch_size=16)
        logging.info("Finish predict emotion in {}".format(time.time()-dep))
        return results

    def summarize(self, inputs, sent_num=3):
        dep = time.time()
        results = []
        for content in inputs:
            results.append(self.content_summarizer(content, num_sentences=sent_num))
        logging.info("Finish summarizing in {}".format(time.time()-dep))
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
        logging.info("Finish classification in {}".format(time.time()-dep))
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
        self.title_process = Search_title_process(args)
        self.content_process = Search_context_process(args)
        self.spider = BaiduSpider()
        self.ini_titles = []
        self.ini_contents = []
    def normal_run(self, keywards, elements_num=args.num):
        filename = './search_results/{}.txt'.format(keywards)
        begin_link = time.time()
        data = self.spider.get_link(keywards, elements_num)
        logging.info("Finish title search in {}".format(time.time()-begin_link))
        for item in data:
            self.ini_titles.append(item['title'][3:])
        begin_title = time.time()
        title_features = self.title_process.pre_processing(self.ini_titles)
        end_title = time.time()
        logging.info("Finish title search in {}".format(begin_title-begin_link))
        data = self.spider.get_content(data)
        end_content = time.time()
        logging.info("Finish title search in {}".format(end_content-end_title))
        for item in data:
            self.ini_contents.append(item['content'])
        content_features = self.content_process.pre_processing(self.ini_contents)
        end_feature = time.time()
        logging.info("Finish title search in {}".format(end_feature-end_content))
        for i, item in enumerate(data):
            title_f = []
            content_f = []
            for key in title_features.keys():
                title_f.append(key + ":{}".format(title_features[key][i]))
            for key in content_features.keys():
                content_f.append(key + ":{}".format(content_features[key][i]))
            with open(filename,'a',encoding='utf-8')as f:
                writing=item['title'] + '\n' + item['link'] + '\n' + item['content'] + '\n'
                for __ in title_f:
                    writing += __ + ' '
                for __ in content_f:
                    writing += __ + '\n'
                f.write(writing+'\n')

if __name__ == "__main__":
    titles = ['瑞士，何必呢？', '发生什么事了？亏了12年的爱奇艺突然赚钱了？',  '卢锋：中国这项数据落后于印度等国，该引起重视']
    contents =  ['''瑞士与中国升级自贸协定的谈判“陷入停滞”，是因为中国“态度不积极”？
    瑞士几家媒体最近两天几乎同时“爆出”这样一则消息。
    报道没有提及官方解释，却援引所谓“专家”分析，说是因为瑞士接连就人权问题对华抹黑引发中方不满。
    这样一番操作下来，立即就有了渲染“中国在搞经济胁迫”的熟悉味道。
    瑞士政府一直希望把与中国的自贸协定更新升级。
    这份协定2011年启动谈判，2014年正式实施，当时是中国与欧洲国家的首个类似文件。
    随后，瑞士延续自上世纪70年代末就开始深度受益于中国改革开放和经济发展的势头，两国的经贸联系持续加深。如今，中国已经成为瑞士第三大贸易伙伴。
    现在，瑞士方面想在升级后的协定中降低更多瑞士商品的关税，并引入“可持续发展方面”的条款。但按瑞士国家经济事务秘书处给媒体的说法，相关谈判过去几年进展缓慢，“北京方面态度不太积极”。
    仅从报道上看，瑞士“官方”透露的信息到此为止。接下来就轮到媒体们“发挥演绎”了。
    瑞士“新苏黎世报周日版”以“中国僵局”为题，“联想”起瑞士过去3年加大对中国人权问题的“批评”。路透社等美欧媒体转载时，也大都突出“中国因此拒绝更新自贸协定”的意思。
    不得不说，瑞士政客与媒体这波操作来得“很是时候”。
    29日是啥时候？是联合国人权高专17年来首次访华之行结束的次日。西方一众反华媒体甚至美国国务院正在各种“质疑”甚至指责巴切莱特女士没就人权问题对华采取强硬立场。
    瑞士国内这波炒作，加上其他一些美欧主流媒体的转载帮腔，无法不让人产生联想：它们是想给借着人权问题抹黑攻击中国添一把柴。
    但其实，连瑞士国内经济界都不赞成政府对着中国鼓噪所谓人权问题。
    瑞士工业协会的经济政策负责人对媒体直言，不断把矛头对准中国，“最终只会导致关系破裂”。
    对于美欧媒体有关中瑞自贸协定的这波炒作，我外交部发言人今天也做出回应。
    主要两层意思：一个显然是对中瑞自贸协定及其升级前景予以肯定。
    发言人说，这份协定自2014年实施以来，给两国和两国人民带来巨大的、实实在在的利益。协定升级有利于双方进一步发掘经贸合作潜力，推动疫情形势下两国经济复苏和发展。
    二呢，则是着重强调了这样几句话：
    中瑞自贸协定是一份互利互惠的协定，不是一方对另一方的恩赐。一直以来，双方本着相互尊重、平等相待的精神，就自贸协定升级保持密切沟通和积极谈判。中方乐见自贸协定升级，同时也希望瑞方能够排除人为干扰因素，同中方相向而行。
    划重点啊，“不是一方对另一方的恩赐”“希望瑞方能够排除人为干扰因素”……
    为什么会有这番话？瑞士那些媒体甚至政府恐怕都该好好咂摸咂摸。''',

    '''「 爱优腾 」这个组合，大家应该不陌生。
    这三兄弟是长视频三巨头，也是亏损三巨头，十年烧掉 1000 多亿，盈利遥遥无期的事迹在江湖一直流传。
    但，现在，这个组合要没了。
    这张表你细品，会发现只有会员服务营收是上升的，广告服务、内容分发收入以及其他收入都在跌。。。
    会员服务的营收为什么会上升呢？
    一来是因为爱奇艺去年涨了一波会员费，单个会员带来的收入变多。
    之前连续包月次月续费是 19 元
    二来是本季度爱奇艺比较集中的释放了一波颇受好评的剧集《 人世间 》《 心居 》《 一年一度喜剧大赛 》，优质的内容让会员人数回暖了一下。
    爱奇艺本季度每会员平均收入为 14.69 元，去年同期为 13.64 元；本季度日均订阅会员数为 1.008 亿，上季度日均订阅会员数是 9640 万人。
    本季度占营收大头的会员服务收入 44.71 亿，同比增长 4% 。
    在线广告服务同比下跌了 30% ，这是因为爱奇艺推出的综艺节目数量减少。
    主要是两点导致的，一来是近几年因为国内综艺节目乱象频发，现在对偶像养成系选秀控制严格，二来是爱奇艺的新战略。
    营收能力变弱了，但却成功扭亏为盈，为啥？
    反应快的差友应该已经想到了：因为成本大幅下降。
    再具体一点就是，靠裁员跟砍内容。
    其中，研究与开发费用为 4.75 亿，同比下降了 29% ，主要是由于与人员相关的薪酬变少。
    销售，一般和行政费用的支出为 7.448 亿，同比下降了 38% ，这项来自营销支出、人事相关薪酬费用等等的减少，另外政府补助了 9080 万。
    '以上都是 “ 小头 ” ，最主要的是收入成本的下降，本季度收入成本为 60 亿，同比下降 16% 。。
    而收入成本下降的关键，在于内容成本，本季度内容成本为 44 亿，同比下降 19% 。
    这波成效主要源自爱奇艺的新内容战略：优化内容成本，提高运营效率。''',

    '''我们知道在2020年初中美两国在经过两年多的贸易战以后，达成了第一阶段的贸易协议，六章的内容。根据能公开得到的信息，总体来看执行还是比较平顺的。但是第六章中中国进口的购买计划，以2020年作为基数，每年还要增加一千亿，也就两年增加两千亿，现在看来没有完全达到这个目标，相差的比例应该是将近1/3左右。
    实际上我们实现了预期的，我讲的这个是货物进口，货物进口实现了4720亿美元的66.5%，但这个背后的原因主要是由于疫情的冲击，以及美国供给能力的制约。美国现在对这个问题公开的信息中，我们还没有看到它有一个非常全面的评估。
    中国跟美国的投资在直接投资方面受到美国限制性政策的影响，我国对美的投资今年显著回落，但是美方对中方的投资是大体稳定的。中概股问题面临一个大的波动，这个我们都关注得很多了，这个背后主要是因为两国对这一类中概股企业信息披露政策的监管要求有一个明显的差距，所以就面临一个波动，这个问题下一步或许还是有可能通过谈判找到一个解决的办法。
    证券投资方面由于没有双边的数据，没法直接看。最近的情况有一些变化，长期来看中国吸引证券投资还是在增加的。
    还有一个旅游和留学情况，这方面的情况主要还不是因为贸易战，贸易战以后两国旅游人数有所减少，特别是新冠疫情爆发以后，双方旅游人数两年中下跌了8到9成，我国赴美的留学人数增量延续了较长时期的回落趋势。
    中国的国际货币地位跟美国作为一个主要国际货币国的地位，要做一个比较的话，有两点特征是非常清楚的。
    第一个中国人民币的国际化指标，总的来讲，过去的几年在持续提升，无论是在储备资产中的占比，还是在国际支付中的占比，这个不仅包括贸易支付，也包括金融交易的支付，都在显著地提升，这个是在波动中提升，在持续相对稳定的提升。
    第二句话，总的来讲这些指标，中国的绝对水平跟美国比较仍然是非常低的。这在提示我们，人民币国际化的前途非常远大，但是任重道远，可能需要长期的渐进式的积累和突破性的推进相结合。
    另外一点，我觉得是特别重要的、令人眼前一亮的亮点，就中国的新能源行业发展得比较快。第一个就风光发电量增长非常快，十年前还不及美国，现在已经远远超过美国了。但这或许也跟欧美国家，对风力发电、光伏发电有了不同的意见或者不同的判断，有一些关系。
    在电动汽车销售的市场上，中国表现是非常突出的。全世界的电动汽车都在进入快速普及的背景下，我们国家的销售量出现了一个井喷式的增长。去年的数据显示，中国电动汽车上牌的数量已经超过了欧盟，并且比欧盟、美国和其他国家加在一起还要多。
    对于平台互联网企业，中国研发的费用也在快速的增加，尽管这个平台企业面临一些困难，但还在增加。但是也要看到跟美国这个平台企业比较，我们仍然有五、六倍的差距，但这里面没有包括华为。
    最后一个问题可能是需要关注的存在问题，中国独角兽的后劲不足。实际上这已经引发了广泛的关注。独角兽企业是我们观察一个国家和地区创新能力、创新活力和生态指标的关键指标之一，也是一个国家产业和经济转型升级的重要推动因素。
    我和我的同事们专门对截止到今年5月份独角兽的统计数据进行观察，可以看到2018年中国的独角兽新增量是41，比美国还多，累计占比在2016年到2017年，接近全世界的50%。
    但是过去的几年中，中国独角兽的量尽管在波动中增长，比如2021年增长到46个，但是跟全世界其他地区比较，比如说跟美国的300多个比较，甚至跟其他国家的171个比较，我们已经相对落后了。
    今年的前四个月，美国是94个，我们只有3个，我看了一下国别的比较，我们国家到了第六位，落后于印度等其他一些国家，到了第六位。我觉得这个背后的原因，可能跟国内企业发展的短期环境条件是不是都有一些关系？我觉得是非常值得关注、重视和应对的。
    最后，从短期的宏观经济的角度，我觉得中国的经济下行压力进一步加大。也就是去年年底，中央经济工作会议提到的三重压力：需求、供给和预期。实际上在今年“两会”以后，出现了下行压力进一步加大的形势。特别是调查失业率又达到了一个就新高，甚至超过了疫情时期的新高。
    特别让我感到有一些意想不到的是，中国青年人就16-24岁的失业率达到了18.2%。我记得几年前我跟我的同事们为G20研究就业问题的时候，看到了欧洲的数据或者一些新兴经济体的数据，年轻人失业率达到了20%，我当时就觉得这个是太难以想象了，面临多大的压力，会出现这样一个情况。现在中国的数据也到了这个水平，我觉得是非常值得重视的。''']
    
    
    # spider = BaiduSpider()
    # spider.search(args.keywords)
    # Search_process = Search_title_process(args)
    # Content_process = Search_context_process(args)
    # begin = time.time()
    # Search_process.dir_output(titles)
    # finish_search = time.time()
    # print("finish in {}".format(finish_search - begin))
    # Content_process.dir_output(contents)
    # print("Finish all content process in {}".format(time.time() - finish_search))
    search_engine = Rational_search(args)
    search_engine.normal_run(args.keywords)
    
    
    
    