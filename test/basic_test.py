import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from splade4elastic import *


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.text = "Coffee is good for you"
        # self.text = "My name is John"

    def test_base_with_ignore(self):
        splade = MLMBaseRewriter("roberta-base", expansions_per_word=3, multi_word="ignore")
        print("Testing MLMBaseRewriter with ignore multi-word option")
        print(splade.query_expand(self.text), end="\n\n")
        self.assertTrue(True)

    def test_base_with_split(self):
        splade = MLMBaseRewriter("roberta-base", expansions_per_word=3, multi_word="split")
        print("Testing MLMBaseRewriter with split multi-word option")
        print(splade.query_expand(self.text), end="\n\n")
        self.assertTrue(True)

    def test_base_with_filter(self):
        splade = MLMBaseRewriter("roberta-base", expansions_per_word=3, multi_word="filter")
        print("Testing MLMBaseRewriter with filter multi-word option")
        print(splade.query_expand(self.text), end="\n\n")
        self.assertTrue(True)

    def test_splade(self):
        splade = SpladeRewriter("roberta-base", expansions_per_word=3)
        print("Testing SpladeRewriter")
        print(splade.query_expand(self.text), end="\n\n")
        self.assertTrue(True)

    def test_linear(self):
        splade = LinearMLMRewriter("roberta-base", expansions_per_word=3)
        print("Testing LinearMLMRewriter")
        print(splade.query_expand(self.text), end="\n\n")
        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
