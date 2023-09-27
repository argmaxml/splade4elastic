import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from splade4elastic import *


class BasicTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_base_with_ignore(self):
        splade = MLMBaseRewriter("roberta-base", expansions_per_word=3, multi_word="ignore")
        text = "Coffee is good for you"
        print(splade.query_expand(text))
        self.assertTrue(True)

    def test_base_with_split(self):
        splade = MLMBaseRewriter("roberta-base", expansions_per_word=3, multi_word="split")
        text = "Coffee is good for you"
        print(splade.query_expand(text))
        self.assertTrue(True)

    def test_splade(self):
        splade = SpladeRewriter("roberta-base", expansions_per_word=3)
        text = "Coffee is good for you"
        print(splade.query_expand(text))
        self.assertTrue(True)

    def test_linear(self):
        splade = LinearMLMRewriter("roberta-base", expansions_per_word=3)
        text = "Coffee is good for you"
        print(splade.query_expand(text))
        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
