## Introduction
H-VDEP (Valuing Defense by Estimating Probabilities in Handball) is a machine learning‐based metric for evaluating team defense in handball. By predicting the probabilities of key defensive events—conceded goals, fouls, and fast breaks—H-VDEP quantitatively assesses the risk and value of each defensive action. This repository provides the code to compute H-VDEP.  
If you have any questions or issues, please contact the author.

## Sample Video
![H-VDEP Sample Video](https://github.com/YOUR_USERNAME/H-VDEP/assets/sample_video_link)  
*Example animation demonstrating the H-VDEP analysis in action.*

## Sample Result
Below is an example of H-VDEP output, illustrating the evaluation of defensive actions during a match.

<div align="left">
<img src="https://github.com/YOUR_USERNAME/H-VDEP/assets/sample_result_image" width="50%" />
</div>

## Author
Ren Kobayashi - kobayashi.ren@g.sp.m.is.nagoya-u.ac.jp
## Requirements
- Python 3.x  
- To install dependencies, run:  
  `pip install -r requirements.txt`

## Evaluation from Scratch

### Step 1: Downloading the Required Data
Please download the necessary datasets from [Google Drive](https://drive.google.com/drive/folders/YOUR_DRIVE_LINK):
- `handball_match_data.xlsx`: Annotation data.
- `tracking_data.json`: Player coordinate data.
- `videos`: Raw match videos (optional for extracting player positions).

### Step 2: Running the Code and Checking the Results
1. Execute the analysis script:  
   `python3 run_H-VDEP.py`
2. Review the generated figures in the `results` folder.
