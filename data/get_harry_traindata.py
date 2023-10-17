def filter_text(line):
    flag = True

    if ('*' in line) and (len(line) <= 8):
        flag = False

    try:
        num = int(line)
        flag = False
    except:
        pass
    
    if 'THE END' in line:
        flag = False

    if (line == '\n') or (line == '\n\n'):
        flag = False

    return flag

all_data = []

for jj in range(1, 4):
    with open(f'HarryPotter/HarryPotter{jj}.txt', 'r') as f:
        data = [i.replace('﹛﹛', '') for i in f.readlines()]

        all_data.extend(data)

all_data = filter(filter_text, all_data)

with open('harrypotter_train_text.txt', 'w') as f:
    f.write(''.join(all_data))

