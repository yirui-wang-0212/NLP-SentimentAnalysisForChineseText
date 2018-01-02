import pickle
import jieba
import os
import re
import string

test_words = []

FindPath = '../../raw_data/test/'
FileNames = os.listdir(FindPath)
for file_name in FileNames:
    full_file_name = os.path.join(FindPath, file_name)
    if 'utf8' in full_file_name:
        with open(full_file_name, 'r', encoding='utf-8') as test_f:
            test_text = test_f.read()
            test_text = ''.join(test_text.split())
            # test_text = re.sub(string.punctuation, "", test_text)
            test_text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", test_text)
            test_list = jieba.cut(test_text, cut_all=False)
            test_words.append(list(test_list))


output = open('../../pkl_data/test/test_review.pkl', 'wb')
pickle.dump(test_words, output)
output.close()
