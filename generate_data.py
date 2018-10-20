import json

with open('data.json', 'rt', encoding='utf-8') as file_reader:
    data = json.load(file_reader)


generate_dict = list()
for i in range(50):
    generate_dict += data.items()

with open('data_.txt', 'wt', encoding='utf-8') as file_writer:
    for item in generate_dict:
        if len(item[0]) > 2:
            file_writer.write(item[0] + '\t' + item[1] + '\n')