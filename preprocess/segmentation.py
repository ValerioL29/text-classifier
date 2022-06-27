import os

import jieba


def stop_words_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_obj:
        lines = file_obj.readlines()
        stopwords = [ele.strip() for ele in lines]

    return stopwords


def seg_sentence(sentence):
    sentence_segmented = jieba.cut(sentence.strip())
    stopwords = stop_words_list('../resources/stop_words_ch.txt')  # 分词词库，这里选择baidu词库
    output_str = ''
    for word in sentence_segmented:
        if word not in stopwords:
            output_str += word
            output_str += "\t"
    return output_str


def segmentation_transform(data_type):
    i = 1
    src_path = '../resources/' + data_type
    for document in os.listdir(src_path):
        target = document.split('.')[0]
        print("Starting Segmentation for {} set: {}-{}\n".format(data_type, i, document))
        output = ''
        with open('{}/{}'.format(src_path, document), 'r+', encoding='utf-8') as file_obj:
            sentences = file_obj.read().split('\n')
            for sentence in sentences:
                res = seg_sentence(sentence)
                if '记者' not in res:
                    output += res
                    output += '\n'

        file_name = '{}/{}.txt'.format(data_type, target)
        with open(file_name, 'w+', encoding='utf-8') as file_obj:
            file_obj.write(output)
        i += 1


# segmentation_transform('train')
segmentation_transform('test')
