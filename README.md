# Deep learning enhanced mixed integer optimization: Learning to reduce model dimensionality 

Niki Triantafyllou, Maria M. Papathanasiou

This is the official implementation of our *Computers & Chemical Engineering* 2024 [paper](https://doi.org/10.1016/j.compchemeng.2024.108725).

## Algorithms

In this repository, you can find the Bayesian optimization algorithm for hyperparameter tuning with the objective of global optimality maximization for:
- Feed-forward neural networks (ANN_bayes.py)
- Convolutional neural networks (CNN_bayes.py)

## Dataset
In this repository, you can find the generated MIP instance (demand profiles) dataset for training and testing (mip_demands.csv), as well as the validation demand profiles as seen in the paper (validation_demands.csv). The CAR T-cell therapy supply chain MIP model (https://www.nature.com/articles/s41598-022-21290-5) is solved for each instance and the MIP dataset is generated. The MIP instances are generated based on Algorithm 1 of the [paper](https://doi.org/10.1016/j.compchemeng.2024.108725).


## Citation
Please cite our paper if you use this code in your work.
```
@article{triantafyllou2024deep,
  title={Deep learning enhanced mixed integer optimization: Learning to reduce model dimensionality},
  author={Triantafyllou, Niki and Papathanasiou, Maria M},
  journal={Computers \& Chemical Engineering},
  volume={187},
  pages={108725},
  year={2024},
  publisher={Elsevier}
}
