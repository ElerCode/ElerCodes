import os
import glob
from multiprocessing import Pool
from functools import partial


def add_class(newdict, filepath):
    filename = filepath.split('/')[-1]
    with open(newdict + filename, "w", encoding="utf-8") as f1:
        with open(filepath, "r+", encoding="utf-8") as f:
            data = f.readlines()
        if data[0].split(' ')[0] == 'class' or data[0].split(' ')[1] == 'class':
            newdata = data
        else:
            methodname = data[0].split('(')[0].split(' ')[-1]
            newdata = ['public class ' + methodname + '{\n']
            newdata.extend(data)
            newdata.append('}')
        for line in newdata:
            f1.writelines(line)


def joern_parse(file, outdir):
    name = file.split('/')[-1].split('.java')[0]
    out = outdir + name + '.bin'
    os.environ['file'] = str(file)
    os.environ['out'] = str(out)
    #print(file,out)
    os.system('/home/user/joern/joern-cli/joern-parse $file --output $out')  # --language c
    #print('bin ok')


def joern_export(bin, outdir):
    name = bin.split('/')[-1].split('.bin')[0]
    out = outdir + name

    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)
    os.system('/home/user/joern/joern-cli/joern-export $bin --repr cfg --out $out')


def get_cfg(inputfile, output_path1, output_path2, type):

    if output_path1[-1] == '/':
        output_path1 = output_path1
    else:
        output_path1 += '/'

    if output_path2[-1] == '/':
        output_path2 = output_path2
    else:
        output_path2 += '/'

    if os.path.exists(output_path1):
        pass
    else:
        os.mkdir(output_path1)

    if os.path.exists(output_path2):
        pass
    else:
        pass
    
    if type == 'parse':
        joern_parse(inputfile, output_path1)
    elif type == 'export':
        name = inputfile.split('/')[-1].split('.java')[0]
        binfile = output_path1 + name + '.bin'
        joern_export(binfile, output_path2)
    else:
        print('Type error!')


def changepath_and_addquotation(path, newpath):
    
    old_name = path + '0-cfg.dot'
    new_name = newpath + path.split('-')[0].split('/')[-1] + '.dot'
    
    print(old_name + '->' + new_name)

    with open(new_name, "w", encoding="utf-8") as f1:
        with open(old_name, "r+", encoding="utf-8") as f:
            data = f.readlines()
        for line in data:
            if "label" in line:
                ls = line.split('label = ')
                line = ''
                line += ls[0]
                line += 'label = '
                line += '\"'
                k = ls[1].split(',')[1]
                if len(ls[1].split(',')) > 2:
                    i = 1
                    k = ''
                    while i < len(ls[1].split(',')):
                        k += ls[1].split(',')[i]
                        k += ','
                        i += 1

                if "<SUB>" in k:
                    line += k.split(')<SUB>')[0]
                else:
                    line += k.split(')>')[0]
                line += '\" ]\n'
            f1.writelines(line)

def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data) returns a list containing the relative paths of all items in the current directory
        file_data = path_data + "/" + i  # Absolute path of all items inside the current folder
        if os.path.isfile(file_data):  # os.path.isfile checks if it's a file. If it's a file, delete it. If it's a folder, recursively call del_file.
            os.remove(file_data)
        else:
            del_file(file_data)
    os.rmdir(path_data)

def cfg_generation(javapath, dotdict):

    newdict =  './'+ javapath.split('/')[-1].split('.')[0]+ 'temp/'
    if os.path.exists(newdict):
        pass
    else:
        os.mkdir(newdict)

    add_class(newdict, javapath)

    bindict = './'+ javapath.split('/')[-1].split('.')[0]+'joern-bin/'
    cfg_tempt = './'+ javapath.split('/')[-1].split('.')[0]+'-temp-cfg/'

    get_cfg(newdict, bindict, cfg_tempt, 'parse')   # Parse the files in newdict and save the results in bindict
    get_cfg(newdict, bindict, cfg_tempt, 'export')  # Export the files in bindict, and save the generated cfg folders in cfg-tempt
    changepath_and_addquotation(cfg_tempt, dotdict)  # Extract 0-cfg from the cfg files in cfg-tempt, remove the redundant parts in labels, and put the results into dotdict

    cfgpath = dotdict + javapath.split('/')[-1].split('.java')[0] + '.dot'

    del_file(newdict)
    del_file(bindict)
    del_file(cfg_tempt)
    return cfgpath

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file)

if __name__ == "__main__":
    dict = './dataset/id2sourcecode/'
    existfile = []
    listdir(dict, existfile)
    for filename in existfile:
        filepath = dict + filename
        try:
            cfg_generation(filepath, "./cfg-dot/")
        except Exception as e:
            print(filepath,e)


