## Agent based modelling for continuously varying supply chains

### The code of the paper `Agent based modelling for continuously varying supply chains` is presented at this repository.

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
### Paper citation

DOI: [10.13140/RG.2.2.31988.68481](https://arxiv.org/abs/2312.15502)

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



### BibTeX

```bibtex
@article{wang2023agent,
  title={Agent based modelling for continuously varying supply chains},
  author={Wang, Wan and Wang, Haiyan and Sobey, Adam J},
  journal={arXiv preprint arXiv:2312.15502},
  year={2023}
}
```
