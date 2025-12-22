import os
import sys
import getopt
from GetSimilarity.GetSimilarity_Token1 import NiCad_Similarity
from GetSimilarity.GetSimilarity_Token2 import SourcererCC_Similarity
from GetSimilarity.GetSimilarity_Token3 import LVMapper_Similarity
from GetSimilarity.GetSimilarity_Tree1 import AST2014
from GetSimilarity.GetSimilarity_Tree2 import BWCCA2015
from GetSimilarity.GetSimilarity_Tree3 import COMPSAC2018
from GetSimilarity.GetSimilarity_Graph1 import StoneDetector
from GetSimilarity.GetSimilarity_Graph2 import Centroids
from GetSimilarity.GetSimilarity_Graph3 import ATVHunter2
import faulthandler

def main(argv):
    tool = ''
    try:
        opts, args = getopt.getopt(argv, "ht:", ["tool="])
    except getopt.GetoptError:
        print('runner.py -t <tools> <sourcefile1> <sourcefile2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('runner.py -t <tools> <sourcefile1> <sourcefile2>\n\
            a1--AST2014 a2--BWCCA2015 a3--COMPSAC2018\n\
            t1--NiCad t2--SourcererCC t3--LVMapper t4--NIL\n\
            c1--ICSME2021 c2--SSEPRW2017\n\
            p1--JSS2022')
            sys.exit()
        elif opt in ("-t", "--tool"):
            tool = arg
    if len(args) < 2:
        print('runner.py -t <tools> <src1> <src2>')

    return tool, args[0], args[1]


def runner(tool, sourcefile1, sourcefile2):
    # token
    if tool == 't1':
        return NiCad_Similarity(sourcefile1, sourcefile2)

    elif tool == 't2':
        return SourcererCC_Similarity(sourcefile1, sourcefile2)

    elif tool == 't3':
        return LVMapper_Similarity(sourcefile1, sourcefile2)

    # ast
    elif tool == 'a1':
        return AST2014(sourcefile1, sourcefile2)

    elif tool == 'a2':
        return BWCCA2015(sourcefile1, sourcefile2)

    elif tool == 'a3':
        return COMPSAC2018(sourcefile1, sourcefile2)

    # cfg
    elif tool == 'c1':
        return StoneDetector(sourcefile1, sourcefile2, 0)

    elif tool == 'c2':
        return Centroids(sourcefile1, sourcefile2)

    elif tool == 'c3':
        return ATVHunter2(sourcefile1, sourcefile2)


