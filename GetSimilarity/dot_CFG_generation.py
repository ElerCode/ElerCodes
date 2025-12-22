#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eler: Ensemble Learning-based Automated Verification of Code Clones
CFG (Control Flow Graph) Generation Module

This script generates Control Flow Graphs (CFG) from Java source files using Joern.
"""

import argparse
import os
import glob
import shutil
from multiprocessing import Pool
from functools import partial


# Default Joern installation path
DEFAULT_JOERN_PATH = '/home/user/joern/joern-cli'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Control Flow Graphs (CFG) from Java source files using Joern.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CFG for a single file
  python dot_CFG_generation.py -i ./code/example.java -o ./cfg-dot/
  
  # Generate CFGs for all files in a directory
  python dot_CFG_generation.py -i ./dataset/id2sourcecode/ -o ./cfg-dot/ --batch
  
  # Specify custom Joern path
  python dot_CFG_generation.py -i ./code/ -o ./cfg/ --joern /path/to/joern-cli
        """
    )
    parser.add_argument(
        '-i', '--input',
        help='Path to Java source file or directory containing Java files',
        required=True,
        type=str
    )
    parser.add_argument(
        '-o', '--output',
        help='Directory to save generated CFG dot files',
        required=True,
        type=str
    )
    parser.add_argument(
        '--joern',
        help=f'Path to Joern CLI directory (default: {DEFAULT_JOERN_PATH})',
        type=str,
        default=DEFAULT_JOERN_PATH
    )
    parser.add_argument(
        '--batch',
        help='Process all Java files in the input directory',
        action='store_true'
    )
    parser.add_argument(
        '--keep-temp',
        help='Keep temporary files (for debugging)',
        action='store_true'
    )
    parser.add_argument(
        '-p', '--processes',
        help='Number of parallel processes for batch mode (default: 1)',
        type=int,
        default=1
    )
    
    return parser.parse_args()


def ensure_dir(path):
    """Ensure directory exists, create if not."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def normalize_path(path):
    """Ensure path ends with /."""
    return path if path.endswith('/') else path + '/'


def add_class_wrapper(input_filepath, output_dir):
    """
    Wrap Java method with a class declaration if needed.
    
    Args:
        input_filepath: Path to input Java file
        output_dir: Directory to save wrapped file
        
    Returns:
        Path to the wrapped file
    """
    filename = os.path.basename(input_filepath)
    output_filepath = os.path.join(output_dir, filename)
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    if not data:
        raise ValueError(f"Empty file: {input_filepath}")
    
    # Check if already has class declaration
    first_line_tokens = data[0].split()
    if len(first_line_tokens) >= 2 and ('class' in first_line_tokens[:2]):
        new_data = data
    else:
        # Extract method name and wrap with class
        method_name = data[0].split('(')[0].split()[-1]
        new_data = [f'public class {method_name} {{\n']
        new_data.extend(data)
        new_data.append('}\n')
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_data)
    
    return output_filepath


def joern_parse(java_file, output_dir, joern_path):
    """
    Parse Java file using Joern to generate binary representation.
    
    Args:
        java_file: Path to Java source file
        output_dir: Directory to save bin file
        joern_path: Path to Joern CLI directory
        
    Returns:
        Path to generated bin file
    """
    name = os.path.basename(java_file).replace('.java', '')
    bin_file = os.path.join(output_dir, f'{name}.bin')
    
    joern_parse_cmd = os.path.join(joern_path, 'joern-parse')
    
    # Use subprocess for better error handling
    cmd = f'{joern_parse_cmd} "{java_file}" --output "{bin_file}"'
    ret = os.system(cmd)
    
    if ret != 0:
        raise RuntimeError(f"Joern parse failed for {java_file}")
    
    return bin_file


def joern_export(bin_file, output_dir, joern_path):
    """
    Export CFG from Joern binary file.
    
    Args:
        bin_file: Path to Joern bin file
        output_dir: Directory to save exported CFG
        joern_path: Path to Joern CLI directory
        
    Returns:
        Path to exported CFG directory
    """
    name = os.path.basename(bin_file).replace('.bin', '')
    cfg_dir = os.path.join(output_dir, name)
    
    joern_export_cmd = os.path.join(joern_path, 'joern-export')
    
    cmd = f'{joern_export_cmd} "{bin_file}" --repr cfg --out "{cfg_dir}"'
    ret = os.system(cmd)
    
    if ret != 0:
        raise RuntimeError(f"Joern export failed for {bin_file}")
    
    return cfg_dir


def process_cfg_dot(cfg_dir, output_dir, original_name):
    """
    Process raw CFG dot file: extract, clean and format.
    
    Args:
        cfg_dir: Directory containing raw CFG files
        output_dir: Directory to save processed dot file
        original_name: Original file name (without extension)
        
    Returns:
        Path to processed dot file
    """
    # Find the 0-cfg.dot file
    raw_dot_file = os.path.join(cfg_dir, '0-cfg.dot')
    
    if not os.path.exists(raw_dot_file):
        # Try to find any cfg.dot file
        dot_files = glob.glob(os.path.join(cfg_dir, '*cfg.dot'))
        if dot_files:
            raw_dot_file = dot_files[0]
        else:
            raise FileNotFoundError(f"No CFG dot file found in {cfg_dir}")
    
    output_file = os.path.join(output_dir, f'{original_name}.dot')
    
    with open(raw_dot_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    processed_lines = []
    for line in data:
        if 'label = ' in line:
            # Parse and clean label
            parts = line.split('label = ')
            prefix = parts[0]
            label_part = parts[1] if len(parts) > 1 else ''
            
            # Extract meaningful label content
            label_content = ','.join(label_part.split(',')[1:])
            
            # Clean up SUB tags and formatting
            if '<SUB>' in label_content:
                label_content = label_content.split(')<SUB>')[0]
            elif ')>' in label_content:
                label_content = label_content.split(')>')[0]
            
            # Reconstruct line with quoted label
            line = f'{prefix}label = "{label_content}" ]\n'
        
        processed_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    return output_file


def remove_directory(path):
    """Safely remove a directory and all its contents."""
    if os.path.exists(path):
        shutil.rmtree(path)


def cfg_generation(java_file, output_dir, joern_path=DEFAULT_JOERN_PATH, keep_temp=False):
    """
    Generate CFG dot file from Java source file.
    
    Args:
        java_file: Path to Java source file
        output_dir: Directory to save generated CFG dot file
        joern_path: Path to Joern CLI directory
        keep_temp: Whether to keep temporary files
        
    Returns:
        Path to generated CFG dot file
    """
    # Get base name
    base_name = os.path.basename(java_file).replace('.java', '')
    
    # Create temporary directories
    temp_base = f'./{base_name}_temp'
    temp_java_dir = f'{temp_base}/java/'
    temp_bin_dir = f'{temp_base}/bin/'
    temp_cfg_dir = f'{temp_base}/cfg/'
    
    ensure_dir(temp_java_dir)
    ensure_dir(temp_bin_dir)
    ensure_dir(temp_cfg_dir)
    ensure_dir(output_dir)
    
    try:
        # Step 1: Wrap with class if needed
        wrapped_file = add_class_wrapper(java_file, temp_java_dir)
        
        # Step 2: Parse with Joern
        bin_file = joern_parse(wrapped_file, temp_bin_dir, joern_path)
        
        # Step 3: Export CFG
        cfg_dir = joern_export(bin_file, temp_cfg_dir, joern_path)
        
        # Step 4: Process and clean dot file
        dot_file = process_cfg_dot(cfg_dir, output_dir, base_name)
        
        return dot_file
        
    finally:
        # Clean up temporary files
        if not keep_temp:
            remove_directory(temp_base)


def list_java_files(directory):
    """
    Recursively list all Java files in directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of Java file paths
    """
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files


def process_single_file(java_file, output_dir, joern_path, keep_temp):
    """
    Process a single Java file (for parallel processing).
    
    Returns:
        Tuple of (filepath, success, result/error)
    """
    try:
        result = cfg_generation(java_file, output_dir, joern_path, keep_temp)
        return (java_file, True, result)
    except Exception as e:
        return (java_file, False, str(e))


def main():
    """Main function."""
    args = parse_args()
    
    # Validate Joern path
    joern_parse_path = os.path.join(args.joern, 'joern-parse')
    if not os.path.exists(joern_parse_path):
        print(f"Error: Joern not found at '{args.joern}'")
        print(f"Please install Joern or specify correct path with --joern")
        return 1
    
    # Ensure output directory exists
    ensure_dir(args.output)
    
    if args.batch or os.path.isdir(args.input):
        # Batch mode: process all Java files in directory
        if not os.path.isdir(args.input):
            print(f"Error: '{args.input}' is not a directory")
            return 1
        
        java_files = list_java_files(args.input)
        print(f"Found {len(java_files)} Java files to process")
        
        success_count = 0
        fail_count = 0
        
        if args.processes > 1:
            # Parallel processing
            process_func = partial(
                process_single_file,
                output_dir=args.output,
                joern_path=args.joern,
                keep_temp=args.keep_temp
            )
            
            with Pool(processes=args.processes) as pool:
                results = pool.map(process_func, java_files)
            
            for filepath, success, result in results:
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"Failed: {filepath} - {result}")
        else:
            # Sequential processing
            for i, java_file in enumerate(java_files):
                print(f"[{i+1}/{len(java_files)}] Processing: {java_file}")
                try:
                    cfg_generation(java_file, args.output, args.joern, args.keep_temp)
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"  Error: {e}")
        
        print(f"\nCompleted: {success_count} success, {fail_count} failed")
        
    else:
        # Single file mode
        if not os.path.isfile(args.input):
            print(f"Error: '{args.input}' is not a file")
            return 1
        
        if not args.input.endswith('.java'):
            print(f"Warning: '{args.input}' does not have .java extension")
        
        print(f"Processing: {args.input}")
        try:
            result = cfg_generation(args.input, args.output, args.joern, args.keep_temp)
            print(f"Generated: {result}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
