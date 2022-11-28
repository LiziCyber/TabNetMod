from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import Tensor
from collections.abc import Sequence
from itertools import chain
from scipy.sparse import issparse
from scipy.sparse.base import spmatrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import torch
import scipy
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp


class TabNetDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.train:
            return self.X[index], self.y[index]
        else:
            return self.X[index]


def create_sampler(weights, train_dataset):
    """
    This creates a sampler from the given weights

    Parameters
    ----------
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    train_dataset : torch.util.data.Dataset
        Training set
    """

    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            y_train = torch.Tensor([item[1] for item in train_dataset]).numpy()
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )

            weights = 1.0 / class_sample_count

            samples_weight = np.array([weights[int(t)] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        y_train = np.array([item[1] for item in train_dataset])
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        y_train = np.array([item[1] for item in train_dataset])
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler


def create_dataloaders(
        train_dataset, eval_datasets, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    train_dataset : torch.util.data.Dataset
        Training data (X, y)
    eval_datasets : list of torch.util.data.Dataset
        List of eval torch.util.data.Dataset (X, y)
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    batch_size : int
        how many samples per batch to load
    num_workers : int
        how many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process
    drop_last : bool
        set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not
        divisible by the batch size, then the last batch will be smaller
    pin_memory : bool
        Whether to pin GPU memory during training

    Returns
    -------
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
        Training and validation dataloaders
    """
    need_shuffle, sampler = create_sampler(weights, train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=need_shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    valid_dataloaders = []
    for eval_dataset in eval_datasets:
        valid_dataloaders.append(
            DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

    return train_dataloader, valid_dataloaders


def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.

    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim)  to performe reduce
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)


def filter_weights(weights):
    """
    This function makes sure that weights are in correct format for
    regression and multitask TabNet

    Parameters
    ----------
    weights : int, dict or list
        Initial weights parameters given by user

    Returns
    -------
    None : This function will only throw an error if format is wrong
    """
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return


def validate_eval_set(eval_datasets, eval_name, train_dataset):
    """Check if the shapes of eval_datasets are compatible with train_dataset.

    Parameters
    ----------
    eval_datasets : list of torch.util.data.Dataset (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    train_dataset : torch.util.data.Dataset (X, y).

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_datasets))]

    assert len(eval_datasets) == len(
        eval_name
    ), "eval_datasets and eval_name have not the same length"

    assert all(
        train_dataset[0][0].shape == eval_dataset[0][0].shape for eval_dataset in eval_datasets
    ), "the shapes of the X from eval_datasets are not compatible with train_dataset"

    '''assert all(
        train_dataset.train == eval_dataset.train for eval_dataset in eval_datasets
    ), "the parameter \'train\' from eval_datasets are not compatible with train_dataset"

    if train_dataset.train:
        assert all(
            train_dataset[0][1].shape == eval_dataset[0][1].shape for eval_dataset in eval_datasets
        ), "the shapes of the y from eval_datasets are not compatible with train_dataset"'''

    return eval_name, eval_datasets


def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)

        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def _assert_all_finite(X, allow_nan=False):
    """Like assert_all_finite, but only for ndarray."""

    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in "fc"
    if is_float and (np.isfinite(np.sum(X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (
                allow_nan
                and np.isinf(X).any()
                or not allow_nan
                and not np.isfinite(X).all()
        ):
            type_err = "infinity" if allow_nan else "NaN, infinity"
            raise ValueError(msg_err.format(type_err, X.dtype))
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype("object") and not allow_nan:
        if np.isnan(X).any():
            raise ValueError("Input contains NaN")


def assert_all_finite(X, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix
    allow_nan : bool
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)


def _unique_multiclass(y):
    if hasattr(y, "__array__"):
        return np.unique(np.asarray(y))
    else:
        return set(y)


def _unique_indicator(y):
    """
    Not implemented
    """
    raise IndexError(
        f"""Given labels are of size {y.shape} while they should be (n_samples,) \n"""
        + """If attempting multilabel classification, try using TabNetMultiTaskClassification """
        + """or TabNetRegressor"""
    )


_FN_UNIQUE_LABELS = {
    "binary": _unique_multiclass,
    "multiclass": _unique_multiclass,
    "multilabel-indicator": _unique_indicator,
}


def unique_labels(*ys):
    """Extract an ordered array of unique labels

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    if not ys:
        raise ValueError("No argument has been passed.")
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

    # Check that we don't mix string type with number type
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))


def _is_integral_float(y):
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)


def is_multilabel(y):
    """Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : numpy array of shape [n_samples]
        Target values.

    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    if hasattr(y, "__array__"):
        y = np.asarray(y)
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
        return (
                len(y.data) == 0
                or np.unique(y.data).size == 1
                and (
                        y.dtype.kind in "biu"
                        or _is_integral_float(np.unique(y.data))  # bool, int, uint
                )
        )
    else:
        labels = np.unique(y)

        return len(labels) < 3 and (
                y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
        )


def check_classification_targets(y):
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
    """
    y_type = type_of_target(y)
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError("Unknown label type: %r" % y_type)


def type_of_target(y):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multiclass-multioutput'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    valid = (
                    isinstance(y, (Sequence, spmatrix)) or hasattr(y, "__array__")
            ) and not isinstance(y, str)

    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), " "got %r" % y
        )

    sparseseries = y.__class__.__name__ == "SparseSeries"
    if sparseseries:
        raise ValueError("y cannot be class 'SparseSeries'.")

    if is_multilabel(y):
        return "multilabel-indicator"

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return "unknown"

    # The old sequence of sequences format
    try:
        if (
                not hasattr(y[0], "__array__")
                and isinstance(y[0], Sequence)
                and not isinstance(y[0], str)
        ):
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (y.dtype == object and len(y) and not isinstance(y.flat[0], str)):
        return "unknown"  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return "unknown"  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # check float and contains non-integer float values
    if y.dtype.kind == "f" and np.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        _assert_all_finite(y)
        return "continuous" + suffix

    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return "multiclass" + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return "binary"  # [1, 2] or [["a"], ["b"]]


def check_unique_type(y):
    target_types = pd.Series(y).map(type).unique()
    if len(target_types) != 1:
        raise TypeError(
            f"Values on the target must have the same type. Target has types {target_types}"
        )


def infer_output_dim(y_train):
    """
    Infer output_dim from targets

    Parameters
    ----------
    y_train : np.array
        Training targets

    Returns
    -------
    output_dim : int
        Number of classes for output
    train_labels : list
        Sorted list of initial classes
    """
    check_unique_type(y_train)
    train_labels = unique_labels(y_train)
    output_dim = len(train_labels)

    return output_dim, train_labels


def check_output_dim(labels, y):
    if y is not None:
        check_unique_type(y)
        valid_labels = unique_labels(y)
        if not set(valid_labels).issubset(set(labels)):
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --
                             contains unkown targets from training --
                             {set(labels)}"""
            )
    return


def infer_multitask_output(y_train):
    """
    Infer output_dim from targets
    This is for multiple tasks.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets

    Returns
    -------
    tasks_dims : list
        Number of classes for output
    tasks_labels : list
        List of sorted list of initial classes
    """

    if len(y_train.shape) < 2:
        raise ValueError(
            "y_train should be of shape (n_examples, n_tasks)"
            + f"but got {y_train.shape}"
        )
    nb_tasks = y_train.shape[1]
    tasks_dims = []
    tasks_labels = []
    for task_idx in range(nb_tasks):
        try:
            output_dim, train_labels = infer_output_dim(y_train[:, task_idx])
            tasks_dims.append(output_dim)
            tasks_labels.append(train_labels)
        except ValueError as err:
            raise ValueError(f"""Error for task {task_idx} : {err}""")
    return tasks_dims, tasks_labels
