# README

NOTE: This version of TabNet is Modified and used in my Bachelor's Thesis.
The original repo is [here](https://github.com/dreamquark-ai/tabnet).
The new model structure is as the graph below. This version has better performance than the original TabNet.

![image](pics/image093.png)

# TabNet : Attentive Interpretable Tabular Learning

This is a pyTorch implementation of Tabnet (Arik, S. O., & Pfister, T. (2019). 
[TabNet: Attentive Interpretable Tabular Learning.](https://arxiv.org/pdf/1908.07442.pdf) 
arXiv preprint arXiv:1908.07442.)

# What problems does pytorch-tabnet handles?

- TabNetClassifier : binary classification and multi-class classification problems
- TabNetRegressor : simple and multi-task regression problems

# How to use it?

TabNet is now scikit-compatible, training a TabNetClassifier or TabNetRegressor is really easy.

```python
from tabnet.model import TabNetClassifier, TabNetRegressor

clf = TabNetClassifier()  #TabNetRegressor()
clf.fit(
  train_dataset,
  eval_set=[valid_dataset]
)
preds = clf.predict(test_dataset)
```

The targets on `y_train/y_valid` should contain a unique type (i.e. they must all be strings or integers).

### Default eval_metric

A few classical evaluation metrics are implemented (see bellow section for custom ones):
- binary classification metrics : 'auc', 'accuracy', 'balanced_accuracy', 'logloss'
- multiclass classification : 'accuracy', 'balanced_accuracy', 'logloss'
- regression: 'mse', 'mae', 'rmse', 'rmsle'


Important Note : 'rmsle' will automatically clip negative predictions to 0, because the model can predict negative values.
In order to match the given scores, you need to use `np.clip(clf.predict(X_predict), a_min=0, a_max=None)` when doing predictions.


### Custom evaluation metrics

It's easy to create a metric that matches your specific need. Here is an example for gini score (note that you need to specifiy whether this metric should be maximized or not):

```python
from tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score

class Gini(Metric):
    def __init__(self):
        self._name = "gini"
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score[:, 1])
        return max(2*auc - 1, 0.)

clf = TabNetClassifier()
clf.fit(
  train_dataset,
  eval_set=[valid_dataset],
  eval_metric=[Gini]
)

```

# Semi-supervised pre-training

Added later to TabNet's original paper, semi-supervised pre-training is now available via the class `TabNetPretrainer`:

```python
# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax' # "sparsemax"
)

train_dataset.train = False
valid_dataset.train = False

unsupervised_model.fit(
    train_dataset=train_dataset,
    eval_datasets=[valid_dataset],
    pretraining_ratio=0.8,
)

clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, # how to use learning rate scheduler
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax' # This will be overwritten if using pretrain model
)

train_dataset.train = True
valid_dataset.train = True

clf.fit(
    train_dataset=train_dataset,
    eval_datasets=[valid_dataset],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    from_unsupervised=unsupervised_model
)
```

The loss function has been normalized to be independent of `pretraining_ratio`, `batch_size` and number of features in the problem.
A self supervised loss greater than 1 means that your model is reconstructing worse than predicting the mean for each feature, a loss bellow 1 means that the model is doing better than predicting the mean.

A complete example can be found within the notebook `pretraining_example.ipynb`.

/!\ : current implementation is trying to reconstruct the original inputs, but Batch Normalization applies a random transformation that can't be deduced by a single line, making the reconstruction harder. Lowering the `batch_size` might make the pretraining easier.



## Model parameters

- `n_d` : int (default=8)

    Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting.
    Values typically range from 8 to 64.

- `n_a`: int (default=8)

    Width of the attention embedding for each mask.
    According to the paper n_d=n_a is usually a good choice. (default=8)

- `n_att`: int (default=8)

    Dimension of the transformed features feeding Attentive Transformer

- `n_steps` : int (default=3)

    Number of steps in the architecture (usually between 3 and 10)  

- `gamma` : float  (default=1.3)

    This is the coefficient for feature reusage in the masks.
    A value close to 1 will make mask selection least correlated between layers.
    Values range from 1.0 to 2.0.

- `cat_idxs` : list of int (default=[] - Mandatory for embeddings) 

    List of categorical features indices.

- `cat_dims` : list of int (default=[] - Mandatory for embeddings)

    List of categorical features number of modalities (number of unique values for a categorical feature)
    /!\ no new modalities can be predicted

- `cat_emb_dim` : list of int (optional)

    List of embeddings size for each categorical features. (default =1)

- `n_independent_att` : int  (default=2)

    Number of independent Gated Linear Units layers at each step.
    Usual values range from 1 to 5.

- `n_shared_att` : int (default=2)

    Number of shared Gated Linear Units at each step
    Usual values range from 1 to 5

- `n_independent_out` : int  (default=2)

    Number of independent Gated Linear Units layers at each step.
    Usual values range from 1 to 5.

- `n_shared_out` : int (default=2)

    Number of shared Gated Linear Units at each step
    Usual values range from 1 to 5

- `epsilon` : float  (default 1e-15)

    Should be left untouched.

- `seed` : int (default=0)

    Random seed for reproducibility

- `momentum` : float

    Momentum for batch normalization, typically ranges from 0.01 to 0.4 (default=0.02)

- `clip_value` : float (default None)

    If a float is given this will clip the gradient at clip_value.
    
- `lambda_sparse` : float (default = 1e-3)

    This is the extra sparsity loss coefficient as proposed in the original paper. The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help.

- `optimizer_fn` : torch.optim (default=torch.optim.Adam)

    Pytorch optimizer function

- `optimizer_params`: dict (default=dict(lr=2e-2))

    Parameters compatible with optimizer_fn used initialize the optimizer. Since we have Adam as our default optimizer, we use this to define the initial learning rate used for training. As mentionned in the original paper, a large initial learning of ```0.02 ```  with decay is a good option.

- `scheduler_fn` : torch.optim.lr_scheduler (default=None)

    Pytorch Scheduler to change learning rates during training.

- `scheduler_params` : dict

    Dictionnary of parameters to apply to the scheduler_fn. Ex : {"gamma": 0.95, "step_size": 10}

- `model_name` : str (default = 'DreamQuarkTabNet')

    Name of the model used for saving in disk, you can customize this to easily retrieve and reuse your trained models.

- `saving_path` : str (default = './')

    Path defining where to save models.

- `verbose` : int (default=1)

    Verbosity for notebooks plots, set to 1 to see every epoch, 0 to get None.

- `device_name` : str (default='auto')
    'cpu' for cpu training, 'gpu' for gpu training, 'auto' to automatically detect gpu.

- `mask_type: str` (default='sparsemax')
    Either "sparsemax" or "entmax" : this is the masking function to use for selecting features

## Fit parameters

- `X_train` : np.array

    Training features

- `y_train` : np.array

    Training targets

- `eval_set`: list of tuple  

    List of eval tuple set (X, y).  
    The last one is used for early stopping  

- `eval_name`: list of str  
              List of eval set names.  

- `eval_metric` : list of str  
              List of evaluation metrics.  
              The last metric is used for early stopping.

- `max_epochs` : int (default = 200)

    Maximum number of epochs for trainng.
    
- `patience` : int (default = 15)

    Number of consecutive epochs without improvement before performing early stopping.

    If patience is set to 0 then no early stopping will be performed.

    Note that if patience is enabled, best weights from best epoch will automatically be loaded at the end of `fit`.

- `weights` : int or dict (default=0)

    /!\ Only for TabNetClassifier
    Sampling parameter
    0 : no sampling
    1 : automated sampling with inverse class occurrences
    dict : keys are classes, values are weights for each class

- `loss_fn` : torch.loss or list of torch.loss

    Loss function for training (default to mse for regression and cross entropy for classification)
    When using TabNetMultiTaskClassifier you can set a list of same length as number of tasks,
    each task will be assigned its own loss function

- `batch_size` : int (default=1024)

    Number of examples per batch, large batch sizes are recommended.

- `virtual_batch_size` : int (default=128)

    Size of the mini batches used for "Ghost Batch Normalization".
    /!\ `virtual_batch_size` should divide `batch_size`

- `num_workers` : int (default=0)

    Number or workers used in torch.utils.data.Dataloader

- `drop_last` : bool (default=False)

    Whether to drop last batch if not complete during training

- `callbacks` : list of callback function  
        List of custom callbacks

- `pretraining_ratio` : float

        /!\ TabNetPretrainer Only : Percentage of input features to mask during pretraining.

        Should be between 0 and 1. The bigger the harder the reconstruction task is.
