import streamlit as st
from modify_search_v2 import Rational_search
from modify_search_v2 import parse_args
import logging
import time
import numpy as np

from PIL import (
    ImageFont,
)

HEADER_INFO = """""".strip()
SIDEBAR_INFO = """
<div class="contributors font-body text-bold">
代码细节请参考
<a class="contributor comma" href="https://github.com/qiuzh20/DL_project">github repo</a>
</div>
"""
PROJECT_INFO = """
<h2 class="font-title">欢迎使用理性搜素引擎 </h2>
<p class="strong font-body">
<span class="d-block extra-info">在这个项目中，我们尝试使用一系列已有的自然语言处理技术，来使得搜索过程更为高效！</span>
</p>
""".strip()

STORY = """
<div class="story-box font-body">

身处数字化时代, “标题党”和“信息茧房”都是引起人们关注的话题。
“标题党”让我们需要点开文章才能获得其相关信息，并且往往会引诱人点开不必要的信息，这些都会浪费我们搜索信息的时间；
“信息茧房”指人在推荐算法的影响下往往只能接触到算法“最希望”人看到的信息，这在带来一定便利性的同时，阻碍了我们充分利用互联网上丰富的信息，甚至会加深人的偏见。
在此情况下，我们希望综合利用现有的NLP工具，开发一款帮助用户高效获得所需信息的工具。

具体来说，这一工具能够：\n
分析并屏蔽标题党（即<strong>标题分析</strong>部分）\n
分析文章内容，返回摘要、关键词、情感倾向等（即<strong>正文处理部分</strong>）\n
根据标题与文章的分析结果，定制输出

</div>
""".strip()


@st.cache(allow_output_mutation=True)
def load_Search_engine(args):
    engine = Rational_search(args)
    return engine

def main():
    st.set_page_config(
        page_title="Rational Search Engine",
        page_icon= ":wink:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    col1, col2 = st.columns([6, 4])
    with col2:
        # st.image(load_image_from_local("asset/images/chef-transformer-transparent.png"), width=300)
        st.markdown(SIDEBAR_INFO, unsafe_allow_html=True)

        with st.expander("为什么我们希望做这个项目?", expanded=True):
            st.markdown(STORY, unsafe_allow_html=True)

        with st.expander("参数选择帮助", expanded=False):
            st.markdown("搜索数量", unsafe_allow_html=True)

        with st.expander("功能实现细节", expanded=False):
            st.markdown("功能细节", unsafe_allow_html=True)

    with col1:
        st.markdown(HEADER_INFO, unsafe_allow_html=True)
        st.markdown(PROJECT_INFO, unsafe_allow_html=True)

        clickbit = False
        title_class = False
        summarize = False
        keywords = False
        context_class = False
        context_emo = False
        raw_number = 200
        max_show = 20
        output_type = False
        rank = False
        filter_method = 0.5
        filter_content = -10
        spider = "百度"

        finish_deploy = False
        st.markdown("让我们首先部署需要的搜索引擎！选择你**希望使用**的搜索功能")
        select_spider = st.checkbox("需要选择使用的搜索引擎")
        if select_spider:
            spider = st.radio("希望使用的搜索引擎",
                         ('百度', '必应'))
        title_process = st.checkbox("标题分析")
        if title_process:
            st.markdown("选择你**希望使用**的标题处理方式")
            clickbit = st.checkbox("处理标题党")
            if clickbit:
                customized = st.checkbox("进一步设置处理程度")
                if customized:
                    filter_method = st.slider("以关键词或结构处理标题党(0为完全参考关键词, 1为完全参考标题语法结构)", 0., 1., 0.5, 0.1)
                    filter_content = st.slider('希望以何种程度过滤标题党(0为原始输出, 数值增大使得过滤增强)', 0, 3, 1)
            title_class = st.checkbox("显示标题类别")
        st.markdown("-----------------")
        context_process = st.checkbox("正文处理")
        if context_process:
            st.markdown("选择你**希望使用**的标题处理方式")
            summarize = st.checkbox("生成摘要")
            if summarize:
                summary_meth = st.radio("选择你需要的摘要生成方式(生成式为根据文本信息重新生成语段, 提取式为摘取原文片段组合)",
                         ('生成式', '提取式'))
            keywords = st.checkbox("生成关键词")
            if summarize:
                context_class = st.checkbox("显示文章类别")
                context_emo = st.checkbox("显示情感倾向")
        if clickbit:
            st.markdown("-----------------")
            rank = st.checkbox("依据标题分析结果排序(优先展示较完整标题)")
        if context_emo:
            output_type = st.checkbox("优先输出负面倾向内容")
        st.markdown("-----------------")
        engine_depoly = st.checkbox("已选好参数，布置搜索引擎！")
        # finish_depoly = False
        if engine_depoly:
            args = parse_args()
            args.device = 'gpu' # 目前只部署了GPU版本的
            logging.basicConfig(filename="./search_logs/{}.log".format(time.time()), filemode="w", 
                                format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                                datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
            logger = logging.getLogger(__name__)
            args.logger = logger
            args.keywords_filter = clickbit
            args.grammer_filter = clickbit
            args.class_filter = title_class
            args.extract_keywords = keywords
            args.emotion_cls = context_emo
            args.spider = spider
            if summarize:
                args.generate_summary = summary_meth == '生成式'
                args.extract_summary = summary_meth == '提取式'
            args.cls = context_class
            begin_time = time.time()
            logging.basicConfig(filename="./search_logs/{}.log".format(time.time()), filemode="w", 
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
            logger = logging.getLogger(__name__)
            logger.fatal(args)
            args.logger = logger
            search_engine = load_Search_engine(args)
            finish_time = time.time()
            st.markdown("已完成搜索引擎部署:sparkles:，用时 {:.2f}s".format(finish_time-begin_time))
            input_keywords = st.text_input("请输入希望搜索的关键词")
            if len(input_keywords) > 0:
                st.write("已输入关键词：{}".format(input_keywords))
                parameter_adjust = st.checkbox("调整搜候选数量或最大显示数")
                if parameter_adjust:
                    raw_number = st.number_input("初始搜索范围", min_value= 50, max_value= 500, value=200)
                    max_show = st.number_input("最大显示数量", min_value= 1, max_value= 500, value=20)
                recipe_button = st.button('开始搜索！')
                finish_deploy = True
            else:
                st.write("您尚未输入关键词")
    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    if finish_deploy and recipe_button:
        finish_process = False
        begin_time = time.time()
        if output_type:
            pairs = search_engine.run(input_keywords, raw_number, 'emo')
            finish_time = time.time()
            st.write("搜索及处理共用时: {:.2f}已优先显示判定为负面倾向的信息。".format(finish_time-begin_time, len(pairs)))
            finish_process = True
        else:
            pairs = search_engine.run(input_keywords, raw_number)
            finish_time = time.time()
            st.write("搜索及处理共用时: {:.2f}".format(finish_time-begin_time, len(pairs)))
            finish_process = True
        text_out = pairs[1]
        pairs = pairs[0]
        if rank:
            def title_process(raw_pairs, ratio=filter_method, threshold=filter_content):
                initial_pairs = raw_pairs
                thresh = ratio*threshold
                weights = []
                for pair in initial_pairs:
                    title_f_w = pair[1]
                    title_s = (1-ratio)*title_f_w['keywords'] + ratio*title_f_w['grammer']
                    if title_s < thresh:
                        pair[0]['title'] = "   不用看这标题了👋"
                    weights.append(title_s)
                weights = np.array(weights) * -1.0
                ranks = np.argsort(weights)
                new_pairs = []
                for idx in ranks:
                    new_pairs.append(initial_pairs[idx])
                return new_pairs
            if output_type:
                pairs = title_process(pairs[0]) + title_process(pairs[1])
            else:
                pairs = title_process(pairs)
            st.write("已按照设置对输出内容排序")
        st.write("共获得 {} 个有效结果".format(len(pairs)))
        if finish_process:
            st.download_button("保存搜索内容", data=text_out, file_name="search_result_for_{}.txt".format(input_keywords))
        for i in range(min(len(pairs), max_show)):
            item = pairs[i][0]
            title_f = pairs[i][1]
            content_f = pairs[i][2]
            if 'class' in title_f.keys():
                title_ = """<p>
                            结果{} <a href="{}">{}</a> 类别：{} (概率 {:.2f})
                            </p>""".format(i+1, item['link'], item['title'][3:], title_f['class'], title_f['confidence'])
                st.markdown(title_, unsafe_allow_html=True)
            else:
                st.markdown("""<p>
                                结果{} <a href="{}">{}</a>
                                </p>""".format(i+1, item['link'], item['title'][3:]), unsafe_allow_html=True)
            if len(content_f) > 0:
                if "cls" in content_f:
                    st.markdown("预测文章为 {} 类别 ({})".format(content_f['cls'], content_f['confidence']))
                for __ in content_f:
                    if __ == "cls":
                        pass
                    if __ == "confidence":
                        pass
                    else:
                        st.text_area("第{}个文本处理结果的 {} 为:".format(i+1, __), content_f[__], height=1)
            expander = st.expander("原始正文内容")
            expander.write(item['content'])

if __name__ == '__main__':
    main()

# refer: https://share.streamlit.io/chef-transformer/chef-transformer/main/app.py