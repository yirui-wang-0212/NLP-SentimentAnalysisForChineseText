import xlwt
import pickle
import itertools
import nltk
import os
import sklearn
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pos_f = '../pkl_data/1000/pos_review.pkl'
neg_f = '../pkl_data/1000/neg_review.pkl'

# 1 提取特征方法
# 1.1 把所有词作为特征
def bag_of_words(words):
    return dict([(word, True) for word in words])


# 1.2 把双词搭配（bigrams）作为特征
def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):

    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词

    return bag_of_words(bigrams)


# 1.3 把所有词和双词搭配一起作为特征
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):

    tuple_words = []
    for i in words:
        temp = (i,)
        tuple_words.append(temp)

    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n) # 使用了卡方统计的方法，选择排名前1000的双

    return bag_of_words(tuple_words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


# 2 特征选择方法
# 2.1 计算出整个语料里面每个词的信息量
# 2.1.1 计算整个语料里面每个词的信息量
def create_word_scores():
    posWords = pickle.load(open(pos_f, 'rb'))
    negWords = pickle.load(open(neg_f, 'rb'))

    posWords = list(itertools.chain(*posWords))  # 把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords))  # 同理

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd["neg"][word] += 1

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores  # 包括了每个词和这个词的信息量


# 2.1.2 计算整个语料里面每个词和双词搭配的信息量
def create_word_bigram_scores():
    posdata = pickle.load(open(pos_f, 'rb'))
    negdata = pickle.load(open(neg_f, 'rb'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams  # 词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd["neg"][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


# 2.2 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words


# 2.3 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
def best_word_features(words):
    # load_data()
    # word_scores = create_word_bigram_scores()
    global best_words
    # best_words = find_best_words(word_scores, 7500)
    return dict([(word, True) for word in words if word in best_words])


pos_review = []  # 积极数据
neg_review = []  # 消极数据


# 3 分割数据及赋予类标签
# 3.1 载入数据
def load_data():
    global pos_review, neg_review
    pos_review = pickle.load(open(pos_f, 'rb'))
    neg_review = pickle.load(open(neg_f, 'rb'))


# 3.2 使积极文本的数量和消极文本的数量一样 (跳过)

# 3.3 赋予类标签
# 3.3.1 积极
def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos_review:
        posWords = [feature_extraction_method(i), 'pos']  # 为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures


# 3.3.2 消极
def neg_features(feature_extraction_method):

    negFeatures = []
    for j in neg_review:
        negWords = [feature_extraction_method(j), 'neg']  # 为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures


train = []  # 训练集
devtest = []  # 开发测试集
test = []  # 测试集
dev = []  #
tag_dev = []


# 3.4 把特征化之后的数据数据分割为开发集和测试集
def cut_data(posFeatures, negFeatures):
    global train, devtest, test
    # train = posFeatures[300:] + negFeatures[300:]
    # devtest = posFeatures[300:500] + negFeatures[300:500]
    # test = posFeatures[:500] + negFeatures[:500]
    train = posFeatures[1500:] + negFeatures[1500:]
    devtest = posFeatures[:500] + negFeatures[:500]


# 4.1 开发测试集分割人工标注的标签和数据
def cut_devtest():
    global dev, tag_dev
    dev, tag_dev = zip(*devtest)


# 4.2 使用训练集训练分类器
# 4.3 用分类器对开发测试集里面的数据进行分类，给出分类预测的标签
# 4.4 对比分类标签和人工标注的差异，计算出准确度
def score(classifier):
    classifier = nltk.SklearnClassifier(classifier)  # 在nltk 中使用scikit-learn的接口
    classifier.train(train)  #训练分类器

    pred = classifier.classify_many(dev)  # 对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_dev, pred)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度


def try_diffirent_classifiers():

    results = list()
    results.append(score(BernoulliNB()))
    results.append(score(MultinomialNB()))
    results.append(score(LogisticRegression()))
    results.append(score(SVC()))
    results.append(score(LinearSVC()))
    results.append(score(NuSVC()))

    return results


best_words = []


# 4.5 检验不同分类器和不同的特征选择的结果
def compare_test():

    global pos_review, neg_review
    classifiers = ['BernoulliNB', 'MultinomiaNB', 'LogisticRegression', 'SVC', 'LinearSVC', 'NuSVC']

    load_data()

    # 创建 xls 文件对象
    wb = xlwt.Workbook()
    # 新增一个表单
    sh = wb.add_sheet('compare')
    col_cnt = 0

    # 使用所有词作为特征
    posFeatures = pos_features(bag_of_words)
    negFeatures = neg_features(bag_of_words)

    cut_data(posFeatures, negFeatures)
    cut_devtest()

    sh.write(0, 0, '所有词')
    col_cnt += 1
    results = try_diffirent_classifiers()
    temp = 0
    for i in classifiers:
        sh.write(col_cnt, 0, i)
        sh.write(col_cnt, 1, results[temp])
        col_cnt += 1
        temp += 1

    # 使用双词搭配作为特征
    posFeatures = pos_features(bigram)
    negFeatures = neg_features(bigram)

    cut_data(posFeatures, negFeatures)
    cut_devtest()

    col_cnt += 1
    sh.write(col_cnt, 0, '双词搭配')
    col_cnt += 1
    results = try_diffirent_classifiers()
    temp = 0
    for i in classifiers:
        sh.write(col_cnt, 0, i)
        sh.write(col_cnt, 1, results[temp])
        col_cnt += 1
        temp += 1

    # 使用所有词加上双词搭配作为特征
    posFeatures = pos_features(bigram_words)
    negFeatures = neg_features(bigram_words)

    cut_data(posFeatures, negFeatures)
    cut_devtest()

    col_cnt += 1
    sh.write(col_cnt, 0, '所有词和双词搭配')
    col_cnt += 1
    results = try_diffirent_classifiers()
    temp = 0
    for i in classifiers:
        sh.write(col_cnt, 0, i)
        sh.write(col_cnt, 1, results[temp])
        col_cnt += 1
        temp += 1

    dimension = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
                 9000]

    row_cnt = 0
    col_cnt += 1
    sh.write(col_cnt, row_cnt, '信息量丰富的所有词')
    row_cnt += 1
    col_cnt += 1

    temp = 0
    temp_col = col_cnt
    for i in classifiers:
        col_cnt += 1
        sh.write(col_cnt, 0, i)
        temp += 1

    # 计算信息量丰富的词，并以此作为分类特征
    word_scores = create_word_scores()
    for d in dimension:
        col_cnt = temp_col
        sh.write(col_cnt, row_cnt, d)
        col_cnt += 1

        global best_words
        best_words = find_best_words(word_scores, int(d))  # 选择信息量最丰富的d个的特征

        posFeatures = pos_features(best_word_features)
        negFeatures = neg_features(best_word_features)

        cut_data(posFeatures, negFeatures)
        cut_devtest()

        results = try_diffirent_classifiers()
        temp = 0
        for i in classifiers:
            sh.write(col_cnt, row_cnt, results[temp])
            col_cnt += 1
            temp += 1
        row_cnt += 1

    row_cnt = 0
    col_cnt += 1
    sh.write(col_cnt, row_cnt, '信息量丰富的所有词和双词搭配')
    row_cnt += 1
    col_cnt += 1

    temp = 0
    temp_col = col_cnt
    for i in classifiers:
        col_cnt += 1
        sh.write(col_cnt, 0, i)
        temp += 1

    # 计算信息量丰富的词，并以此作为分类特征
    word_scores = create_word_bigram_scores()
    for d in dimension:
        col_cnt = temp_col
        sh.write(col_cnt, row_cnt, d)
        col_cnt += 1

        best_words = find_best_words(word_scores, int(d))  # 选择信息量最丰富的d个的特征

        posFeatures = pos_features(best_word_features)
        negFeatures = neg_features(best_word_features)

        cut_data(posFeatures, negFeatures)
        cut_devtest()

        results = try_diffirent_classifiers()
        temp = 0
        for i in classifiers:
            sh.write(col_cnt, row_cnt, results[temp])
            col_cnt += 1
            temp += 1
        row_cnt += 1

    # 保存文件
    wb.save('../out/compare.xls')
    # word_scores_1 = create_word_scores()
    # word_scores_2 = create_word_bigram_scores()
    # best_words_1 = find_best_words(word_scores_1, 5000)
    # best_words_2 = find_best_words(word_scores_2, 5000)
    # load_data()
    # posFeatures = pos_features(best_word_features, best_words_2)  # 使用所有词作为特征
    # negFeatures = neg_features(best_word_features, best_words_2)
    # cut_data(posFeatures, negFeatures)
    # cut_devtest()
    # # posFeatures = pos_features(bigram)
    # # negFeatures = neg_features(bigram)
    #
    # # posFeatures = pos_features(bigram_words)
    # # negFeatures = neg_features(bigram_words)
    #
    # print('BernoulliNB`s accuracy is %f' % score(BernoulliNB()))
    # print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB()))
    # print('LogisticRegression`s accuracy is %f' % score(LogisticRegression()))
    # print('SVC`s accuracy is %f' % score(SVC()))
    # print('LinearSVC`s accuracy is %f' % score(LinearSVC()))
    # print('NuSVC`s accuracy is %f' % score(NuSVC()))


# 5.1 使用测试集测试分类器的最终效果
def use_the_best():
    word_scores = create_word_bigram_scores()  # 使用词和双词搭配作为特征
    best_words = find_best_words(word_scores, 4000)  # 特征维度1500
    load_data()
    posFeatures = pos_features(best_word_features, best_words)
    negFeatures = neg_features(best_word_features, best_words)
    cut_data(posFeatures, negFeatures)
    trainSet = posFeatures[1500:] + negFeatures[1500:]  # 使用了更多数据
    testSet = posFeatures[:500] + negFeatures[:500]
    test, tag_test = zip(*testSet)


    # 5.2 存储分类器
    def final_score(classifier):
        classifier = SklearnClassifier(classifier)
        classifier.train(trainSet)
        pred = classifier.classify_many(test)
        return accuracy_score(tag_test, pred)

    print(final_score(MultinomialNB()))  #使用开发集中得出的最佳分类器


# 5.3 把分类器存储下来（存储分类器和前面没有区别，只是使用了更多的训练数据以便分类器更为准确）
def store_classifier():
    load_data()
    word_scores = create_word_bigram_scores()
    global best_words
    best_words = find_best_words(word_scores, 7500)

    posFeatures = pos_features(best_word_features)
    negFeatures = neg_features(best_word_features)

    trainSet = posFeatures + negFeatures

    MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
    MultinomialNB_classifier.train(trainSet)
    pickle.dump(MultinomialNB_classifier, open('../out/classifier.pkl', 'wb'))


# 6 使用分类器进行分类，并给出概率值
# 6.1 把文本变为特征表示的形式
def transfer_text_to_moto():

    moto = pickle.load(open('../pkl_data/test/test_review.pkl', 'rb'))  # 载入文本数据

    def extract_features(data):
        feat = []
        for i in data:
            feat.append(best_word_features(i))
        return feat

    moto_features = extract_features(moto)  # 把文本转化为特征表示的形式
    return moto_features


# 6.2 对文本进行分类，给出概率值
def application(moto_features):
    clf = pickle.load(open('../out/classifier.pkl', 'rb'))  # 载入分类器

    pred = clf.prob_classify_many(moto_features)  # 该方法是计算分类概率值的
    p_file = open('../out/test_result.txt', 'w')  # 把结果写入文档
    for i in pred:
        p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
    p_file.close()

if __name__ == '__main__':

    # try:
    #     clf = pickle.load(open('../classifer.pkl', 'rb'))
    # except FileNotFoundError:
    # store_classifier()

    # try:
    #     best_words = pickle.load(open('best_words.pkl', 'rb'))
    # except FileNotFoundError:
    #     print('start creating best_words')
    # load_data()
    # word_scores = create_word_bigram_scores()
    # best_words = find_best_words(word_scores, 7500)
    # output = open('best_words.pkl', 'wb')
    # pickle.dump(best_words, output)
    # output.close()
    #     print('end creating best_words')
    #
    store_classifier()
    # best_words = pickle.load(open('../out/best_words.pkl', 'rb'))
    moto_features = transfer_text_to_moto()
    application(moto_features)


