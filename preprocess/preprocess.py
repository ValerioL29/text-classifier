from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import Bunch
import pynlpir
import pickle
import json
import os


def stop_words_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_obj:
        lines = file_obj.readlines()
        stopwords = [ele.strip() for ele in lines]
    return stopwords


def segmentation_transform(stop_words, category, json_file_path, bunch_obj):
    pynlpir.open()
    with open(json_file_path, 'r+', encoding='utf-8') as file_obj:
        objects = json.load(file_obj)
        for obj in objects:
            content = obj['content']
            title = obj['title']
            try:
                segments = pynlpir.segment(content, pos_english=True)  # 分词函数,返回一个列表
            except RuntimeError:
                print('Oops, something goes wrong')
                pass
            else:
                # 对分词结果取名词
                seg_only_noun = [ele[0] for ele in segments
                                 if ele[1] == 'noun' and ele[0] not in stop_words]
                document_cleaned = '\t'.join(seg_only_noun)
                bunch_obj.filenames.append(title)
                bunch_obj.label.append(category)
                bunch_obj.contents.append(document_cleaned)
    pynlpir.close()
    return


def clean(raw_data_input_path, word_bag_filepath):
    """
    函数说明: 对下载的源新闻文档进行分词处理，并且从分词结果中仅取名词
    :param raw_data_input_path: 源文档根目录，其一级子目录存储各个类别的新闻文章，目录名为类别名
    :param word_bag_filepath: 将清洗的结果bunch持久化到词袋文件
    :return:
    """
    # 定义Bunch的结构
    bunch = Bunch(target_name=[],
                  label=[],
                  filenames=[],
                  contents=[])

    documents = os.listdir(raw_data_input_path)
    categories = [document[:-5] for document in documents]
    bunch.target_name.extend(categories)

    stop_words = stop_words_list('../resources/stop_words_ch.txt')
    for i in range(len(categories)):
        path = raw_data_input_path + '/' + documents[i]
        print("Current documents: {}\n".format(path))
        segmentation_transform(stop_words, categories[i], path, bunch)

    return bunch


train_bunch = clean('../resources/train', 'train/train_bag.pickle')
test_bunch = clean('../resources/test', 'test/test_bag.pickle')

'''
这里使用了max_df和min_df两个参数指定文档频率的上下界，df表示document frequency
即 如果一个词在10%的文档中都出现(本文共10各类别)，那么该词语不能很好的代表某一个类，所以应该将这个词去掉。min_df同理，如果某个词的频率出现太低，则也无法代表某一类文档，应当忽略。
'''
train_tf_idf = TfidfVectorizer(sublinear_tf=True,
                               max_df=0.1,
                               min_df=0.001)
train_tfidf_space = Bunch(target_name=train_bunch.target_name,
                          label=train_bunch.label,
                          filenames=train_bunch.filenames,
                          tfidf_weight_matrices=[],
                          vocabulary={})
test_tfidf_space = Bunch(target_name=test_bunch.target_name,
                         label=test_bunch.label,
                         filenames=test_bunch.filenames,
                         tfidf_weight_matrices=[],
                         vocabulary={})

train_tfidf_space.tfidf_weight_matrices = train_tf_idf.fit_transform(train_bunch.contents)
train_tfidf_space.vocabulary = train_tf_idf.vocabulary_

test_tf_idf = TfidfVectorizer(sublinear_tf=True,
                              max_df=0.1,
                              min_df=0.001,
                              vocabulary=train_tfidf_space.vocabulary)
test_tfidf_space.tfidf_weight_matrices = test_tf_idf.fit_transform(test_bunch.contents)
test_tfidf_space.vocabulary = test_tf_idf.vocabulary_

with open('train/train_bag.pickle', 'wb') as file_obj:
    pickle.dump(train_tfidf_space, file_obj)
with open('test/test_bag.pickle', 'wb') as file_obj:
    pickle.dump(test_tfidf_space, file_obj)
