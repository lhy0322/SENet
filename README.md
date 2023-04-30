# SENet
SENet: a deep learning framework for discriminating super- and typical enhancers by sequence information


![Image browser window](figure.png)
## 1. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). We applied training on a single NVIDIA TITAN X with 12 GB graphic memory. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.


#### 1.1 Create and activate a new virtual environment

```
conda create -n senet python=3.8
conda activate senet
```



#### 1.2 Install the package and other requirements

(Required)

```
git clone https://github.com/lhy0322/SENet
cd SENet
python -m pip install -r requirements.txt
```


## 2. Step by step for training model
### Step 1
Use "crossvalid.py" file to pick the hyperparameters of SENet
- *python crossvalid.py* 

### Step 2
Use "train.py" to train SENet model
- *python train.py*

### Step 3
Use "predict.py" to predict super-enhancers
- *python predict.py*
