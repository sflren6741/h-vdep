## Introduction
H-VDEP (Valuing Defense by Estimating Probabilities in Handball) is a machine learning‐based metric for evaluating team defense in handball. By predicting the probabilities of key defensive events—conceded goals, fouls, and fast breaks—H-VDEP quantitatively assesses the risk and value of each defensive action. This repository provides the code to compute H-VDEP.  
If you have any questions or issues, please contact the author.

## Sample Video
![H-VDEP Sample Video](https://github.com/user-attachments/assets/fb3da330-36dd-4130-adcf-33e58f531d01)  
*Example animation demonstrating the H-VDEP analysis in action.*

## Sample Result
1. **H-VDEP Output Result**  
   This image shows the output metrics produced by the H-VDEP model when applied to a match.

<div align="left">
<img src="https://github.com/user-attachments/assets/6b103159-dab6-407e-ac26-17c935ee2e2c" width="50%" />
</div>

2. **Feature Importance**  
   This figure illustrates the importance of various features used in predicting defensive events.

<div align="left">
<img src="https://github.com/user-attachments/assets/211d312d-30d0-4bd7-8be6-059d3fe8a7fb" width="50%" />
</div>

3. **Relationship between Conceded Goals and H-VDEP**  
   This graph depicts the correlation between the number of conceded goals and the H-VDEP score.

<div align="left">
<img src="https://github.com/user-attachments/assets/9959b6e1-c4bc-408f-aa81-069100d20665" width="50%" />
</div>

## Author
Ren Kobayashi - kobayashi.ren@g.sp.m.is.nagoya-u.ac.jp
## Requirements
- Python 3.8.10
- To install dependencies, run:  
  `pip install -r requirements.txt`

## Evaluation from Scratch

### Step 1: Downloading the Required Data
H-VDEP has been evaluated using the Events in Invasion Games Dataset – Handball (EIGD-H). This dataset contains broadcast video streams, synchronized official positional data, and human event annotations from Handball-Bundesliga matches (season 2019/20).  
**License:** The position and video data are provided by [Kinexon](https://kinexon.com/) with authorization from the [Handball-Bundesliga](https://www.liquimoly-hbl.de/en/), and the dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please ensure appropriate credit is given when using this data.  
Download the dataset manually from [https://data.uni-hannover.de/dataset/eigd](https://data.uni-hannover.de/dataset/eigd) or via the provided download script (if available).

### Step 2: Running the Code and Checking the Results
1. Execute the analysis script by running:  
   `python3 main.py`
2. Review the generated figures in the `fig` folder.
