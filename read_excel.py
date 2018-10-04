import pandas as pd
import re

FILE_PATH = 'trys.xlsx'
FILE_NAME = 'resource/ground_truth_data.txt'
IS_APPEND = False

def read_excel(path):
    df = pd.read_excel(path, header=None)
    full_text = list(df[0])
    write_file(FILE_NAME, full_text)
    return full_text

def write_file(file_name, full_text):
    global  IS_APPEND
    a_w = 'a' if IS_APPEND else 'wt'
    file_writer = open(file_name, a_w, encoding='utf-8')
    token_list = list()
    for text in full_text:
        text = str(text)
        text = re.sub('・', '\u3000', text)
        text = re.sub('\s+', '\u3000', text)
        text = re.sub('、', '\u3000', text)
        text = re.sub(' ', '\u3000', text)
        token_list += text.split('\u3000')
    for token in token_list:
        if len(token.strip()) > 1 and not token.strip().isdigit():
            file_writer.write(str(token) + '\n')
    file_writer.close()

# read_excel(FILE_PATH)

def process_file():
    full_text = open(FILE_NAME, 'rt', encoding='utf-8').read().splitlines()
    token_list = list()
    for text in full_text:
        text = re.sub('\d*\.\d*', '', text)
        text = re.sub('\d*n|s\d*', '', text)
        text = re.sub('♯|＃|#|,|', '', text)
        text = re.sub('\d*Kg', '', text)
        text = re.sub('\d*～\d*', '', text)
        text = re.sub('\d{2}-\d{1}', '', text)
        text = re.sub('1', '１', text)
        text = re.sub('2', '２', text)
        text = re.sub('3', '３', text)
        text = re.sub('4', '４', text)
        text = re.sub('5', '５', text)
        text = re.sub('6', '６', text)
        text = re.sub('7', '７', text)
        text = re.sub('8', '８', text)
        text = re.sub('9', '９', text)
        text = re.sub('0', '０', text)
        text = re.sub('R', 'Ｒ', text)
        text = re.sub('H', 'Ｈ', text)
        text = re.sub('P', 'Ｐ', text)
        text = re.sub('X', 'Ｘ', text)
        text = re.sub('S', 'Ｓ', text)
        text = re.sub('C', 'Ｃ', text)
        text = re.sub('M', 'Ｍ', text)
        text = re.sub('k|K', 'ｋ', text)
        text = re.sub('g|G', 'g', text)
        text = re.sub('m', 'ｍ', text)
        text = re.sub('=', '＝', text)
        text = re.sub('＝\d+', '＝', text)
        text = re.sub('Ｘ\d*', '', text)
        text = re.sub('Ｙ\d*', '', text)
        text = re.sub('\d*Ｓ|Ｎ', '', text)
        text = re.sub('^\d+ｋg$', '', text)
        text = re.sub('\*', '', text)
        text = re.sub('Y\d*', '', text)
        text = re.sub('X\d*', '', text)
        text = re.sub('\d{3,}', '', text)
        text = re.sub('\d{3,}', '', text)
        text = re.sub('O', 'Ｏ', text)
        text = re.sub('㎏', 'ｋg', text)
        text = re.sub('\d*ｎ', '', text)
        text = re.sub('^(\d*-\d*)$', '', text)
        text = re.sub('\d+', '#', text)
        text = re.sub('A', 'Ａ', text)
        text = re.sub('B', 'Ｂ', text)
        text = re.sub('F', 'Ｆ', text)
        text = re.sub('L', 'Ｌ', text)
        text = re.sub('N', 'Ｎ', text)
        text = re.sub('Q', 'Ｑ', text)
        text = re.sub('T', 'Ｔ', text)
        text = re.sub('g', 'ｇ', text)
        text = re.sub('r', 'ｒ', text)
        text = re.sub('\)', '', text)
        text = re.sub('\(', '', text)
        text = re.sub('一', 'ー', text)

        if len(text) > 2:
            token_list.append(text)
    write_file(FILE_NAME, set(sorted(token_list)))


process_file()

# def generate_suggest():
#     file_reader = open(FILE_NAME, 'rt', encoding='utf-8')
#     full_text = sorted(set(file_reader.read().splitlines()))
#     file_reader.close()
#
#     file_writer = open('suggest.txt', 'w', encoding='utf-8')
#     for text in full_text:
#         file_writer.write(text + '\n')
#     file_writer.close()





# generate_suggest()
# process_file()