#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import List
import numpy as np


def email_features(word_indices: List[int]) -> np.ndarray:
    """Convert a list of word IDs into a feature vector.

    :param word_indices: a list of word IDs
    :return: a feature vector from the word indices (a row vector)
    """
    
    # Total number of words in the dictionary
    n_words = 1899

    vec_words = np.ndarray((1, n_words), dtype='int');

    for i in range(n_words):
        vec_words[0, i] = 0;

    for i in word_indices:
        vec_words[0, i - 1] = 1;
        
    return vec_words;

    # =========================== END OF YOUR CODE ============================
