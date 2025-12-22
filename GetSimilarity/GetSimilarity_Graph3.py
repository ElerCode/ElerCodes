'''
Convert functions into a long sequence
Directly call the getSimilarity function
input: dot_file_path1, dot_file_path2, bool (whether to normalize)
output: similarity, between 0 and 1
'''

from xmlrpc.client import boolean
import networkx as nx
import os
from simhash import Simhash
import javalang
import xlwt
import sys
import pickle
def getGraph(dot_path):
    return nx.DiGraph(nx.nx_pydot.read_dot(dot_path))

def get_simhash(text):
    # 使用simhash生成指纹
    return str(Simhash(text).value)

def normalizingLine(s): # e.g. "30:  Attribute createAttribute(KeyValuePair kvp)"
    try:
        tokens = list(javalang.tokenizer.tokenize(s[1:-1]))
        temp_line = []   
        for i in range(len(tokens)):
            token = tokens[i]

            token_type = str(type(token)).split('.')[-1][:-2] # -2 is to remove the trailing ['>]
            temp_line.append(get_simhash(token_type)) 
        return temp_line
    except:
        return [get_simhash(s[1:-1])]


def commonLine(s): # without normalization
    try:
        tokens = list(javalang.tokenizer.tokenize(s[1:-1])) # [1:-1] is to remove the leading and trailing quotes
        temp_line = []   
        for i in range(len(tokens)):
            token = tokens[i]
            temp_line.append(get_simhash(token.value))
        return temp_line
    except:
        print(s[1:-1], "can't parse!!!")
        return [get_simhash(s[1:-1])]

def get_root_node(G):
    node = None
    for n in G.nodes(data=True):
        predecessors = G.predecessors(n[0])
        if len(list(predecessors)) == 0:
            node = n
            break
    return node[0]


# Use digraph as input
def getPreOrder(dg, isNorm : bool):
    stack = []
    result = []
    vis = [False for _ in range(10000)] # avoid infinite loop, assuming max node index doesn't exceed 10000
    root = get_root_node(dg)
    stack.append(root)
    while stack:
        curr = stack.pop()
        vis[int(curr)] = True
        # print(dg.nodes[curr]['label'])
        if (isNorm):
            line = normalizingLine(dg.nodes[curr]['label'])
        else:
            line = commonLine(dg.nodes[curr]['label'])
        result.extend(line)
        for nbr in list(dg[curr])[::-1]:
            if(not vis[int(nbr)]):
                stack.append(nbr)
    return result

# Levenshtein edit distance
def editDistance(word1, word2):
    n = len(word1)
    m = len(word2)
    
    # One of the strings is empty
    if n * m == 0:
        return n + m
    
    # DP array
    D = [ [0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize boundary states
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    
    # Calculate all DP values
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1] 
            if word1[i - 1] != word2[j - 1]:
                hamming_distance = Simhash(int(word1[i-1])).distance(Simhash(int(word2[j-1])))
                similarity = 1 - (hamming_distance / 64.0)
                left_down += 1 - similarity     
            D[i][j] = min(left, down, left_down)
    
    return D[n][m]

 
def getSimilarity(dot_path1, dot_path2, isNorm: bool):
    dg1 = getGraph(dot_path1)
    dg2 = getGraph(dot_path2)
    preOrder1 = getPreOrder(dg1, isNorm)
    preOrder2 = getPreOrder(dg2, isNorm)

    edit_dis = editDistance(preOrder1, preOrder2)
    return (1 - edit_dis/max(len(preOrder1), len(preOrder2))) #,size

def ATVHunter2(sourcefile1,sourcefile2):
    dotfile1 = './cfg-dot/'+ sourcefile1.split('/')[-1].split('.')[0] + '.dot'
    dotfile2 = './cfg-dot/'+ sourcefile2.split('/')[-1].split('.')[0] + '.dot'
    return getSimilarity(dotfile1,dotfile2,True)

if __name__ == '__main__':
    root_path = 'progex/dot/'
    dot_files = os.listdir('progex/dot/')
    num_of_file = len(dot_files)
    s = [[0 for _ in range(num_of_file)] for _ in range(num_of_file)]
    for i in range(num_of_file):
        for j in range(i+1, num_of_file):
            sim = getSimilarity(os.path.join(root_path ,dot_files[i]), os.path.join(root_path ,dot_files[j]), False)
            s[i][j] = sim
            # print(sim)

    workbook = xlwt.Workbook(encoding = 'utf-8')        # Create a workbook with utf-8 encoding
    worksheet1 = workbook.add_sheet("filter_sheet")     # Add a new sheet
    worksheet2 = workbook.add_sheet("verify_sheet")

    for i in range(num_of_file):
        worksheet1.write(0,i,dot_files[i])
        worksheet2.write(0,i,dot_files[i])

    for i in range(len(s)):
        for j in range(len(s[i])) :                            
            worksheet1.write(i+1,j,s[i][j])
            worksheet2.write(i+1,j,s[i][j])
    workbook.save('ATVHunter_not_norm_inline.xls') # Note: file format must be xls, not xlsx, otherwise it will throw an error