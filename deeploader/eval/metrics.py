from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

class BinaryClassificationMetrics(object):
    """
    Private container class for classification metric statistics. True/false positive and
     true/false negative counts are sufficient statistics for various classification metrics.
    This class provides the machinery to track those statistics across mini-batches of
    (label, prediction) pairs.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives = 0

    def update_binary_stats(self, label, pred):
        """
        Update various binary classification counts for a single (label, pred)
        pair.

        Parameters
        ----------
        label : `NDArray`
            The labels of the data.

        pred : `NDArray`
            Predicted values.
        """
        if not isinstance(pred, numpy.ndarray):
            from tensorboardX.x2num import make_np
            pred = make_np(pred)
            label = make_np(label)

        pred_label = numpy.argmax(pred, axis=1)
        # check_label_shapes(label, pred)
        if len(numpy.unique(label)) > 2:
            raise ValueError("%s currently only supports binary classification."
                             % self.__class__.__name__)
        pred_true = (pred_label == 1)
        pred_false = 1 - pred_true
        label_true = (label == 1)
        label_false = 1 - label_true

        self.true_positives += (pred_true * label_true).sum()
        self.false_positives += (pred_true * label_false).sum()
        self.false_negatives += (pred_false * label_true).sum()
        self.true_negatives += (pred_false * label_false).sum()

    @property
    def precision(self):
        if self.true_positives + self.false_positives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_positives)
        else:
            return 0.

    @property
    def accuracy(self):
        total = self.total_examples
        if total > 0:
            return float(self.true_positives + self.true_negatives) / total
        else:
            return 0.

    @property
    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_negatives)
        else:
            return 0.

    @property
    def fscore(self):
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0.

    @property
    def matthewscc(self):
        """
        Calculate the Matthew's Correlation Coefficent
        """
        if not self.total_examples:
            return 0.

        true_pos = float(self.true_positives)
        false_pos = float(self.false_positives)
        false_neg = float(self.false_negatives)
        true_neg = float(self.true_negatives)
        terms = [(true_pos + false_pos),
                 (true_pos + false_neg),
                 (true_neg + false_pos),
                 (true_neg + false_neg)]
        denom = 1.
        for t in filter(lambda t: t != 0., terms):
            denom *= t
        return ((true_pos * true_neg) - (false_pos * false_neg)) / math.sqrt(denom)

    @property
    def total_examples(self):
        return self.false_negatives + self.false_positives + \
               self.true_negatives + self.true_positives

    def reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0

    def update(self, label, pred):
        self.update_binary_stats(label, pred)

    def reset(self):
        self.reset_stats()

    def get(self):
        names = ['acc', 'precision', 'recall', 'f1score',
                 'pr', 'necall']
        npos = self.true_positives+self.false_negatives
        nneg = self.true_negatives+self.false_positives
        pr = npos/(npos+nneg)
        necall = self.true_negatives/nneg
        values = [self.accuracy, self.precision, self.recall, self.fscore,
                  pr, necall]
        return names, values

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))