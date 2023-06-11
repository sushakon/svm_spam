#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
from typing import Dict


def get_vocabulary_dict() -> Dict[int, str]:
    """Read the fixed vocabulary list from the datafile and return.

    :return: a dictionary of words mapped to their indexes
    """

    vocab_dict = {}

    with open('data/vocab.txt') as vocab_file:
      words = csv.reader(vocab_file,  delimiter='\t')
      for word in words:
           vocab_dict[int(word[0])] = word[1].strip()

    return vocab_dict;
