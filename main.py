#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eler: Ensemble Learning-based Automated Verification of Code Clones
Feature Extraction Pipeline

This script extracts 9-dimensional similarity features from code pairs
using different clone detection algorithms (token, tree, graph based).
"""

import argparse
import os
import pandas as pd
import sys
from Extraction_of_features import runner
import time
import multiprocessing as mp
import tqdm
import math
from functools import partial


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Eler Feature Extraction - Extract similarity features from code pairs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i ./dataset/BCB_clone.csv -s ./data/id2sourcecode/ -o ./output/
  python main.py --input ./dataset/T1.csv --source ./data/ --output ./output/
        """
        
    )
    parser.add_argument(
        '-i', '--input',
        help='Path to CSV file containing clone pairs (columns: FunID1, FunID2)',
        required=True,
        type=str
    )
    parser.add_argument(
        '-s', '--source',
        help='Directory containing Java source files (named as {id}.java)',
        required=True,
        type=str
    )
    parser.add_argument(
        '-o', '--output',
        help='Directory to save extracted feature CSV files',
        required=True,
        type=str
    )
    parser.add_argument(
        '-p', '--processes',
        help='Number of parallel processes (default: 1, use -1 for all CPUs)',
        type=int,
        default=1
    )
    parser.add_argument(
        '--error-log',
        help='Path to error log file (default: ./errorlog.txt)',
        type=str,
        default='./errorlog.txt'
    )
    parser.add_argument(
        '--parse-error-file',
        help='Path to file containing IDs of files that failed to parse',
        type=str,
        default='./parser_error.txt'
    )
    
    return parser.parse_args()


def get_sim(tool, dataframe, inputpath, wrongfile, logfile):
    """
    Calculate similarity scores for code pairs using specified tool.
    
    Args:
        tool: Tool identifier (t1, t2, t3, a1, a2, a3, c1, c2, c3)
        dataframe: DataFrame containing code pairs
        inputpath: Path to source code directory
        wrongfile: List of file IDs that failed to parse
        logfile: Log file handle for errors
    
    Returns:
        List of similarity scores
    """
    sim = []
    for _, pair in tqdm.tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"{tool}", leave=False):
        id1, id2 = pair.FunID1, pair.FunID2
        
        # Skip graph-based algorithms for files that failed to parse
        if tool[0] == 'c':  # CFG-based algorithms
            if str(id1) in wrongfile or str(id2) in wrongfile:
                sim.append('parse_error')
                continue
        
        sourcefile1 = os.path.join(inputpath, str(id1) + '.java')
        sourcefile2 = os.path.join(inputpath, str(id2) + '.java')
        
        try:
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log = "\n" + time.asctime() + "\t" + tool + "\t" + str(id1) + "\t" + str(id2) + "\t" + similarity
            if logfile:
                logfile.writelines(log)
            similarity = 'False'
        
        sim.append(similarity)
    
    return sim


def cut_df(df, n):
    """Split DataFrame into n parts for parallel processing."""
    df_num = len(df)
    every_epoch_num = math.floor((df_num / n))
    df_split = []
    for index in range(n):
        if index < n - 1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_split.append(df_tem)
    return df_split


def main():
    """Main function for feature extraction."""
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load wrongfile list
    wrongfile = []
    if os.path.exists(args.parse_error_file):
        with open(args.parse_error_file, 'r') as f:
            wrongfile = f.read().split()
    
    # Open log file
    logfile = open(args.error_log, 'a')
    
    # Determine clone type from filename
    inputcsv = args.input
    clone_type = os.path.basename(inputcsv).split('.')[0]
    if 'noclone' in inputcsv.lower() or 'nonclone' in inputcsv.lower():
        clone_type = 'nonclone'
    
    print(f"Processing: {inputcsv}")
    print(f"Clone type: {clone_type}")
    print(f"Source directory: {args.source}")
    print(f"Output directory: {args.output}")
    
    # Read input CSV
    pairs = pd.read_csv(inputcsv, header=None)
    
    # Check if first row is header
    if pairs.iloc[0, 0] in ['FunID1', 'FUNCTION_ID_ONE', 'id1']:
        pairs = pairs.drop(labels=0)
    
    pairs.columns = ['FunID1', 'FunID2'] + list(pairs.columns[2:])
    
    print(f"Total pairs to process: {len(pairs)}")
    
    # Calculate similarity scores using all 9 algorithms
    tools = ['t1', 't2', 't3', 'a1', 'a2', 'a3', 'c1', 'c2', 'c3']
    tool_names = ['NiCad', 'SourcererCC', 'LVMapper', 
                  'AST2014', 'BWCCA2015', 'COMPSAC2018',
                  'StoneDetector', 'GroupDroid', 'ATVHunter']
    
    similarities = {}
    
    for tool, name in zip(tools, tool_names):
        print(f"\nExtracting features with {name} ({tool})...")
        sim = get_sim(tool, pairs, args.source, wrongfile, logfile)
        similarities[f'{tool}_sim'] = sim
        print(f"  Completed: {len(sim)} pairs processed")
    
    # Build result DataFrame
    result = pd.DataFrame({
        'FunID1': pairs['FunID1'].to_list(),
        'FunID2': pairs['FunID2'].to_list(),
        **similarities
    })
    
    # Save results
    output_file = os.path.join(args.output, f'{clone_type}_sim.csv')
    result.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    logfile.close()


if __name__ == '__main__':
    main()
