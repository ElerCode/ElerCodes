import os
import sys
import javalang
from javalang.ast import Node
from anytree import AnyNode, RenderTree,PreOrderIter
import time
import pickle


nodetypedict = {
    'MethodDeclaration': 0,
    'Modifier': 1,
    'FormalParameter': 2,
    'ReferenceType': 3,
    'BasicType': 4,
    'LocalVariableDeclaration': 5,
    'VariableDeclarator': 6,
    'MemberReference': 7,
    'ArraySelector': 8,
    'Literal': 9,
    'BinaryOperation': 10,
    'TernaryExpression': 11,
    'IfStatement': 12,
    'BlockStatement': 13,
    'StatementExpression': 14,
    'Assignment': 15,
    'MethodInvocation': 16,
    'Cast': 17,
    'ForStatement': 18,
    'ForControl': 19,
    'VariableDeclaration': 20,
    'TryStatement': 21,
    'ClassCreator': 22,
    'CatchClause': 23,
    'CatchClauseParameter': 24,
    'ThrowStatement': 25,
    'WhileStatement': 26,
    'ArrayInitializer': 27,
    'ReturnStatement': 28,
    'Annotation': 29,
    'SwitchStatement': 30,
    'SwitchStatementCase': 31,
    'ArrayCreator': 32,
    'This': 33,
    'ConstructorDeclaration': 34,
    'TypeArgument': 35,
    'EnhancedForControl': 36,
    'SuperMethodInvocation': 37,
    'SynchronizedStatement': 38,
    'DoStatement': 39,
    'InnerClassCreator': 40,
    'ExplicitConstructorInvocation': 41,
    'BreakStatement': 42,
    'ClassReference': 43,
    'SuperConstructorInvocation': 44,
    'ElementValuePair': 45,
    'AssertStatement': 46,
    'ElementArrayValue': 47,
    'TypeParameter': 48,
    'FieldDeclaration': 49,
    'SuperMemberReference': 50,
    'ContinueStatement': 51,
    'ClassDeclaration': 52,
    'TryResource': 53,
    'MethodReference': 54,
    'LambdaExpression': 55,
    'InferredFormalParameter': 56
}


# Get the data needed for AST, recursively traverse each node to build a tree
def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))


def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
    else:
        newnode = AnyNode(id=id, token=token, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)

def traversal(node,typedict):  # Recursively traverse all nodes

    if node.children:
        for child in node.children:
            traversal(child,typedict)
    else:  # Traversed to leaf node, output parent nodes upwards in order
        try:
            node.token = typedict[node.token]
        except KeyError:
            if node.token not in nodetypedict:
                node.token = 'String'
        while node.parent:
            node = node.parent

# Code data preprocessing
def Buildast(programfile):
    programtext = programfile.read()
    programtokens = javalang.tokenizer.tokenize(programtext)
    parser = javalang.parse.Parser(programtokens)
    programast = parser.parse_member_declaration()
    tree = programast
    nodelist = []
    newtree = AnyNode(id = 0,token=None)
   
    tokens = list(javalang.tokenizer.tokenize(programfile.read()))
    createtree(newtree, tree, nodelist)

    typedict = {}
    for token in tokens:
        token_type = str(type(token))[:-2].split(".")[-1]
        token_value = token.value
        if token_value not in typedict:
            typedict[token_value] = token_type
        else:
            if typedict[token_value] != token_type:
                print('Need to check at here.')
    traversal(newtree,typedict)
    return newtree



def getNodenum(root):
    if root is None:
        return 0
    elif root.is_leaf:
        return 1
    else:
        res = 1
        for ch in root.children:
            res += getNodenum(ch)
        return res

def gettreesize(root):
    if root is None:
        return  0
    elif root.is_leaf:
        return sys.getsizeof(root)
    else:
        size = sys.getsizeof(root)
        for ch in root.children:
            size += gettreesize(ch)
        return size

def savetree(root,file):
    if root is None:
        return
    elif root.is_leaf:
        pickle.dump(root,file)
        return
    else:
        pickle.dump(root,file)
        for ch in root.children:
            savetree(ch,file)

def Hashforast(tree):
    tmptree = tree
    pass


def Treematch(tree1, tree2):
    """
    Calculate the number of common nodes
    """
    if tree1 is None or tree2 is None:
        return 0

    token1 = tree1.__dict__['token']
    token2 = tree2.__dict__['token']
    if token1 != token2:
        return 0

    ch_a = [x for x in tree1.children]
    ch_b = [x for x in tree2.children]
    m = len(ch_a)
    n = len(ch_b)
    # Use dynamic programming to calculate the maximum number of matching nodes
    res_m = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            res_m[i][j] = max(
                res_m[i - 1][j], res_m[i][j - 1],
                res_m[i - 1][j - 1] + Treematch(ch_a[i - 1], ch_b[j - 1]))
    return res_m[m][n] + 1


def AST2014(f1,f2):
    file1 = open(f1)
    file2 = open(f2)
    tree1, tree2 = Buildast(file1), Buildast(file2)
    file1.close()
    file2.close()
    commonnodes = Treematch(tree1, tree2)
    similarity = 2 * commonnodes / (getNodenum(tree1) + getNodenum(tree2))

    return similarity #, size