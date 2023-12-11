def read_to_txt(file_path):
    filedata = []
    file = open(file_path, 'r', encoding='utf-8')
    while True:
        line = file.readline()
        if not line: break
        line = line.replace('\n', '')
        if line != '': filedata.append(line)
        print(line)
    file.close()
    return filedata

