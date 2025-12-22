# Eler: Ensemble Learning-based Automated Verification of Code Clones

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#usage">Usage</a>
</p>


## Overview

**Eler** is an automated high-precision clone verification approach leveraging ensemble learning techniques. 
It addresses the challenge of verifying whether clone pairs identified by clone detection algorithms are indeed authentic clone pairs.

<img width="950" alt="image" src="./img/method.png">

### Key Features

- **Multi-representation Feature Extraction**: Combines nine clone detection algorithms based on different code representations (token, tree, graph) to extract comprehensive features.
- **Ensemble Learning**: Integrates 11 machine learning classifiers with majority voting to achieve robust verification results.
- **High Precision**: Achieves **95% precision** in clone verification while reducing manual labor by **84.75%**.
- **Multi-type Clone Support**: Effectively validates Type-1, Type-2, Type-3, and Type-4 clones.



## Installation


### Dependencies

```
numpy
pandas
scikit-learn
xgboost
javalang
tqdm
joblib
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/ElerCode/ElerCodes.git
cd ElerCode

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
Eler/
├── main.py                      # Main entry point
├── Extraction_of_features.py    # Implement the Extraction of features phase
├── Ensemble_learning.py         # Implement the Ensemble learning phase
├── GetSimilarity/               # Clone detection algorithms, including three token-based, three tree-based, and three graph-based code clone detection algorithms we reproduced
├── dataset/                     # Data
```

## Usage

### Step 0: CFG Generation

Generate Control Flow Graphs from Java source files using Joern (required for graph-based similarity algorithms).
Note: This step can be skipped — we provide pre-generated CFG .dot files in the GetSimilarity/cfg_dot/ folder.

```bash
python GetSimilarity/dot_CFG_generation.py \
    --input <java_file_or_directory> \
    --output <output_directory> \
    --joern <path_to_joern_cli>
```

**Arguments:**

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `-i, --input` | Path to Java file or directory | Yes | - |
| `-o, --output` | Directory to save CFG dot files | Yes | - |
| `--joern` | Path to Joern CLI directory | No | `/home/user/joern/joern-cli` |
| `--batch` | Process all Java files in directory | No | False |
| `--keep-temp` | Keep temporary files for debugging | No | False |

### Step 1: Feature Extraction

Extract nine-dimensional similarity features from code pairs using different clone detection algorithms.

```bash
python main.py \
    --input <path_to_clone_pairs_csv> \
    --source <path_to_source_code_directory> \
    --output <output_directory>
```

**Arguments:**

| Argument | Description | Required |
|----------|-------------|----------|
| `-i, --input` | Path to CSV file containing clone pairs (FunID1, FunID2) | Yes |
| `-s, --source` | Directory containing Java source files (named as `{id}.java`) | Yes |
| `-o, --output` | Directory to save extracted feature CSV files | Yes |


### Step 2: Training & Evaluation

Train the ensemble learning model and evaluate clone verification performance.

```bash
python Ensemble_learning.py \
    --dir <path_to_feature_csv_directory> \
    --output <output_directory> \
    --mode <train|predict|evaluate> \
    --threshold <voting_threshold>
```

**Arguments:**

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `-d, --dir` | Directory containing feature CSV files | Yes | - |
| `-o, --output` | Directory to save trained models and results | Yes | - |
| `-m, --mode` | Operation mode: `train`, `predict`, or `evaluate` | No | `evaluate` |
| `-t, --threshold` | Voting threshold n (1-11) | No | 6 |
| `--model_dir` | Directory containing pre-trained models (for predict mode) | No | `./models/` |

