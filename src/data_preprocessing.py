"""Build preprocessing components for model-ready car data.

This module is reserved for defining the target, numeric and categorical
feature groups, train/test splitting, and the scikit-learn preprocessing
pipeline. The pipeline will impute, scale, and encode values without fitting
transformations on the test set.
"""
