For more information see the list of the requirments (You can install them `pip install -r requirements.txt`). 
The `main.py` is the file to call to start the training.
The code works with `Python3.7` and `Python3.7-Python3.11`. 

``` Bash
# create conda environment
conda create -n lasc python==3.7.1
conda activate lasc
pip install torch
```
Note that this code does work with TensorFlow 2+. 

### How to Run
For "stochastics supply chain environment", you can start the training with
```
python main.py --env1
python main.py --env2
python main.py --env3
```

For "Batch  supply chain environment", you can start the training with
```
python main.py --env1
python main.py --env2
python main.py --env3
```

