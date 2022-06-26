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
        if word not in stopwords and word != ' ':
            output_str += word
            output_str += "\t"
    return output_str


def transform():
    dataset = [
        '../resources/1_caijing.txt',
        '../resources/2_gupiao.txt',
        '../resources/3_jiaoyu.txt',
        '../resources/4_keji.txt',
        '../resources/5_shehui.txt',
        '../resources/6_shishang.txt',
        '../resources/7_shizheng.txt',
        '../resources/8_tiyu.txt',
        '../resources/9_youxi.txt',
        '../resources/10_yule.txt'
    ]

    for i in range(10):
        print("Starting Segmentation for dataset: {}\n".format(i + 1))
        output = ''
        with open(dataset[i], 'r+', encoding='utf-8') as file_obj:
            sentences = file_obj.read().split('\n')
            for sentence in sentences:
                res = seg_sentence(sentence)
                output += res

        file_name = 'output/{}.txt'.format(i + 1)
        with open(file_name, 'w+', encoding='utf-8') as file_obj:
            file_obj.write(output)
