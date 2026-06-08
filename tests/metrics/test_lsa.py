# -*- coding: utf-8 -*-
"""Behavioral tests for metrics in text_metrics/metrics/lsa.py."""

import pytest

import text_metrics
from text_metrics.metrics.lsa import LsaSentenceAllStd


def lsa_all_std(text):
    return LsaSentenceAllStd().value_for_text(text_metrics.Text(text))


# ---------------------------------------------------------------------------
# lsa_all_std = standard deviation of LSA similarity over ALL sentence pairs
# in the text — computed text-wide, independent of paragraph grouping.
# ---------------------------------------------------------------------------

class TestLsaSentenceAllStd:

    _ONE_PARAGRAPH = (
        "Os professores reuniram-se na sala. "
        "Eles discutiram o currículo novo. "
        "A reunião terminou cedo."
    )
    _SPLIT_2_1 = (
        "Os professores reuniram-se na sala. "
        "Eles discutiram o currículo novo.\n\n"
        "A reunião terminou cedo."
    )
    _ONE_PER_PARAGRAPH = (
        "Os professores reuniram-se na sala.\n\n"
        "Eles discutiram o currículo novo.\n\n"
        "A reunião terminou cedo."
    )

    def test_independent_of_paragraph_grouping(self):
        # The same three sentences yield the same value no matter how they are
        # split into paragraphs — all pairs are considered text-wide.
        baseline = lsa_all_std(self._ONE_PARAGRAPH)
        assert lsa_all_std(self._SPLIT_2_1) == pytest.approx(baseline)
        assert lsa_all_std(self._ONE_PER_PARAGRAPH) == pytest.approx(baseline)

    def test_nonzero_when_every_paragraph_has_one_sentence(self):
        # With one sentence per paragraph there are no within-paragraph pairs;
        # the spread must still come from the cross-paragraph pairs, not zero.
        assert lsa_all_std(self._ONE_PER_PARAGRAPH) > 0.0
