

with open('data/HarryPotter1-7.txt', 'r', encoding='big5') as f:
    data = f.readlines()


st = 0
ed = 0
nn = 1
for index, line in enumerate(data):
    if '.Harry Potter and' == line[1:18] and index > 1:
        print(index, line)
        st = ed
        ed = index

        contents = data[st:ed]
        with open(f'data/HarryPotter{nn}.txt', 'w') as f:
            f.write(''.join(contents))
   
        nn += 1

