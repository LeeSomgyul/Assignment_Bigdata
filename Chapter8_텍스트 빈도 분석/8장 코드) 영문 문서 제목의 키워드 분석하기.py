import pandas as pd
import glob
import re
from functools import reduce
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

all_files = glob.glob('myCabinetExcelData*.xls')

all_files_data = []
for file in all_files:
    data_frame = pd.read_excel(file)
    all_files_data.append(data_frame)
all_files_data[0]

all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)

all_files_data_concat.to_csv('riss_bigdata.csv', encoding = 'utf-8', index = False)

all_title = all_files_data_concat['제목']

stopwords_list = stopwords.words('english')
lemma = WordNetLemmatizer()

stopwords_list = stopwords.words('english')

words = []

for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title))
    EnWordsToken = word_tokenize(EnWords.lower())
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopwords_list]
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]
    words.append(EnWordsTokenStopLemma)
    
words2 = list(reduce(lambda x, y: x+y, words))

count = Counter(words2)

word_count = dict()

for tag, counts in count.most_common(50):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

