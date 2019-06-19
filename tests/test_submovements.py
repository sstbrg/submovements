#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `submovements` package."""

import pytest


from submovements import submovements
from DataProcessing import Preprocessor, Trial
import numpy as np

def test_preprocessor_instance():
    pproc = Preprocessor()
    assert isinstance(pproc, Preprocessor)

def test_trial_instance():
    trial = Trial()
    assert isinstance(trial, Trial)

def test_invalid_source_dir():
    with pytest.raises(AssertionError) as AE:
        pproc = Preprocessor()
        gen = pproc.load_df_from_directory_gen('$')
    assert 'Destination directory does not exist' in str(AE)


