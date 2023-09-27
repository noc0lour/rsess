"""
Tests for the ESS class
"""

import pyrsess
import numpy as np


def test_encode():
    ess = pyrsess.ESS(28, 4, 8)
    sequence = ess.encode([0, 1, 0, 1])
    assert len(sequence) == 4


def test_multi_encode():
    ess = pyrsess.ESS(28, 4, 8)
    sequences = ess.multi_encode(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
    )
    assert sequences.shape == (4, 4)


def test_construction_reference_encode():
    ess = pyrsess.ESS(28, 4, 8)
    nums = np.arange(16)
    bin_nums = ((nums.reshape(-1, 1) & (2 ** np.flip(np.arange(4)))) != 0).astype(int)
    sequences = ess.multi_encode(bin_nums)
    assert (
        sequences
        == np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 3],
                [1, 1, 1, 5],
                [1, 1, 3, 1],
                [1, 1, 3, 3],
                [1, 1, 5, 1],
                [1, 3, 1, 1],
                [1, 3, 1, 3],
                [1, 3, 3, 1],
                [1, 3, 3, 3],
                [1, 5, 1, 1],
                [3, 1, 1, 1],
                [3, 1, 1, 3],
                [3, 1, 3, 1],
                [3, 1, 3, 3],
                [3, 3, 1, 1],
            ]
        )
    ).all()


def test_construction_reference_decode():
    ess = pyrsess.ESS(28, 4, 8)
    nums = np.arange(16)
    bin_nums = ((nums.reshape(-1, 1) & (2 ** np.flip(np.arange(4)))) != 0).astype(int)
    sequences = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 3],
            [1, 1, 1, 5],
            [1, 1, 3, 1],
            [1, 1, 3, 3],
            [1, 1, 5, 1],
            [1, 3, 1, 1],
            [1, 3, 1, 3],
            [1, 3, 3, 1],
            [1, 3, 3, 3],
            [1, 5, 1, 1],
            [3, 1, 1, 1],
            [3, 1, 1, 3],
            [3, 1, 3, 1],
            [3, 1, 3, 3],
            [3, 3, 1, 1],
        ]
    )
    bits = ess.multi_decode(sequences)
    assert (bits == bin_nums).all()
