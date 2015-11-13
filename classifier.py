import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

ROW_LIMIT = 10000


class CBClassifier(object):
    def __init__(self, feature_data, target, classifier="random_forest", classifier_kwargs={}, row_limit=ROW_LIMIT, k_folds=5, scoring='accuracy'):
        """
        We can tune the following parameters:
        - classifier we use (random forest, etc.)
        - number of folds in our standard k-fold cross-validation
        - number of rows of our data to use
        - scoring ('accuracy', 'f1', etc, all options here:
            http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
        """
        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs
        self.row_limit = row_limit
        self.feature_data = feature_data
        self.target = target
        self.k_folds = k_folds
        self.scoring= scoring

    def get_classifier(self):
        return {
            "random_forest": RandomForestClassifier,
            "logistic": LogisticRegression,

        }[self.classifier]

    def classify(self):
        """
        Run the classifier.
        """
        start = time.time()
        classifier = self.get_classifier()(**self.classifier_kwargs)
        scores = cross_validation.cross_val_score(
            classifier,
            self.feature_data[:self.row_limit],
            self.target[:self.row_limit],
            cv=self.k_folds,
            scoring=self.scoring,
        )
        end = time.time()
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "rows": self.row_limit,
            "k_folds": self.k_folds,
            "duration": end - start
        }


