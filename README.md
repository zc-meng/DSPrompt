## Code for "DSPrompt: Prompt Learning with Relation Abstraction and Context Injection for Distant Supervised Relation Extraction" (ECAI 2024)

### 1. Environment Setup
```
pip install -r requirements.txt
```

### 2. Datasets
* We present results on three widely used DSRE datasets: NYT-10d, NYT-10m, Wiki-20m.
```
sh download_nyt10.sh
sh download_nyt10m.sh
sh download_wiki20m.sh
```


### 3. Training and testing models

* Training scripts are provided in the topmost directory for each of the four datasets. Once the training finishes, the best saved model would automatically be tested on the test set (returning AUC, Macro F1, Micro F1, and P@M)
```
sh train_nyt10d.sh
sh train_nyt10m.sh
sh train_wiki20m.sh
```

```
sh infer_nyt10d.sh
sh infer_nyt10m.sh
sh infer_wiki20m.sh
```


### Cite
If you use or extend our work, please cite the following paper:
```
TBD
```

### Acknowledgements
Our codebase is built upon [OpenNRE's](https://aclanthology.org/D19-3029.pdf). For more details on the format of the dataset's used, we refer the user to their [repository](https://github.com/thunlp/OpenNRE).
