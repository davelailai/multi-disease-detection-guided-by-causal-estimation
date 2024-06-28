# multi-disease-detection-guided-by-causal-estimation

## Installation

Below are quick steps for installation:

```shell
git clone https://github.com/davelailai/multi-disease-detection-guided-by-causal-estimation.git
cd multi-disease-detection-guided-by-causal-estimation
pip install -e .
```

train our model as follow:

```shell
bash tools/dist_train.sh configs/FFA/FFA_Q2LCausal.py 1
```

the FFA dataset need to be required by send email to Jianyang.Xie@liverpool.ac.uk