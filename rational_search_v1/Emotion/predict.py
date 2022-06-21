# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from Emotion.utils import convert_example

def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        example = {"text": text}
        input_ids, token_type_ids = convert_example(
            example,
            tokenizer,
            max_seq_length=256,
            is_test=True)
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    confidence = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        confidence.extend(paddle.max(probs, axis=1).tolist())
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results, confidence

class emotion_predictor(object):
    def __init__(self,
                 params_path="/Emotion/checkpoints/model_800/model_state.pdparams",
                 model_name='ernie-3.0-base-zh'):
        self.label_map = {0: 'negative', 1: 'positive'}
        self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_classes=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.set_dict(paddle.load(os.getcwd()+params_path))
    
    def emo_cls(self, data, batch_size=32):
        results = predict(self.model, data, self.tokenizer, self.label_map, batch_size=batch_size)
        return results

if __name__ == "__main__":
    paddle.set_device('gpu')

    data = [
        '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    
    predictor = emotion_predictor()
    results = predictor.emo_cls(data=data)
    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {}, confidence: {}'.format(text, results[0][idx], results[1][idx]))
