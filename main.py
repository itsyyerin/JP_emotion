import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


import preprocess_nlp
import predict_nlp
import test_nlp
import train_nlp

def main():
    # preprocess_nlp.py 파일에서 정의된 함수나 클래스를 사용하여 전처리 작업을 수행합니다
    preprocess_nlp.do_preprocessing()
    # 또는 다른 함수를 호출할 수 있습니다
    # preprocess_nlp.some_other_function()

if __name__ == "__main__":
    main()