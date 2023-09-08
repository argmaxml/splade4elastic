import itertools, collections, json, string, re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List
import numpy as np


class MLMBaseRewriter:
    """Elastic SPLADE model"""

    def __init__(self, model_name: str, expansions_per_word:int = 10):
        """Initialize the model

        Args:
            model_name (str): name of the model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token="<s>")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.k=expansions_per_word
        self.const_weight = 1

    def __tokenize_to_words(self, sentence):
            return sentence.translate(
                {ord(c): " " for c in string.punctuation}
            ).split()
    
    def __tokenize_and_preserve(self, sentence, text_labels=None):
        if type(sentence) == str:
            sentence = self.__tokenize_to_words(sentence)
        if text_labels is None:
            text_labels = itertools.count()
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)


            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
        cnt = itertools.count()
        return [
            (k, [(next(cnt), t, self.tokenizer.convert_tokens_to_ids(t)) for i, t in g])
            for k, g in itertools.groupby(
                zip(labels, tokenized_sentence), lambda x: x[0]
            )
        ]

    def do_expansion(self, word):
        return True
    
    def mask_expansion(self, txt):
        ret = collections.defaultdict(list)
        X = self.tokenizer.encode(txt, return_tensors="pt")
        word2token = self.__tokenize_and_preserve(txt)
        words = self.__tokenize_to_words(txt)

        for wi, lst in word2token:
            if not self.do_expansion(words[wi]):
                # skip this word
                ret[wi].append((words[wi], self.const_weight))
                continue
            X_m = X.clone()
            for mask_token_index, token, _ in lst:
                ti = mask_token_index
                # if self.tokenizer.bos_token:
                #     ti += 1
                X_m[0, ti] = self.tokenizer.mask_token_id
            logits = self.model(X_m).logits
            for mask_token_index, token, _ in lst:
                mask_token_logits = logits[0, mask_token_index, :]
                max_ids = np.argsort(mask_token_logits.to("cpu").detach().numpy())[
                    ::-1
                ][:self.k]
                max_tokens = self.tokenizer.convert_ids_to_tokens(max_ids)
                max_scores = np.sort(mask_token_logits.to("cpu").detach().numpy())[::-1][ :self.k]

                ret[wi].extend(zip(max_tokens, max_scores))
        ret = dict(ret)
        if self.tokenizer.bos_token:
            del ret[0]
        return list(ret.values())

    def __only_alpha(self, txt):
        return "".join(c for c in txt if c in string.ascii_letters)
    
    def logits2weights(self, word_logits):
            return word_logits
    
    def __force_weights(self, word_logits, txt):
        ret = [(w[0], self.const_weight) if w[0] in txt else w for w in word_logits]
        return ret
    
    def __elastic_format(self, expanded_list, text):
        ret = []
        text = text.lower().split()
        for words in expanded_list:
            words = self.logits2weights(words)
            # The original word should have a higher score
            words = self.__force_weights(words, text)
            words = [(self.__only_alpha(w[0]).lower(), w[1]) for w in words if w[0] != self.tokenizer.bos_token]
            # unite equal words and sum their scores
            unique_words = {w[0] for w in words}
            words = [(unique, sum(w[1] for w in words if w[0] == unique)) for unique in unique_words]
            # sort by score
            words = sorted(words, key=lambda x: x[1], reverse=True)
            # print(words)
            or_statement = []
            for w in words:
                or_statement.append(f"{w[0]}^{round(float(w[1]), 2)}")
            or_statement = " OR ".join(or_statement)
            or_statement = f"({or_statement})"
            ret.append(or_statement)
        return " ".join(ret)

    def query_expand(self, text: str):
        """Expands a query using Masked-language-model
        Args:
            text (str): input text
        Returns:
            str: splade-ified text
        """
        me = self.mask_expansion(text)
        return self.__elastic_format(me, text)
    
    def transform(self, X: List[str]):
        """Transforms a list of queries to expanded queries
        Args:
            X (List[str]): List of queries
        Returns:
            List[str]: List of expanded queries
        """
        # TODO: optimize for batch
        return [self.query_expand(t) for t in X]


class LinearMLMRewriter(MLMBaseRewriter):
    def logits2weights(self, word_logits):
            min_score = min(w[1] for w in word_logits)
            ret = [(w[0], w[1] - min_score) for w in word_logits]
            norm_factor = sum(w[1] for w in ret)
            ret = [(w[0], w[1]/norm_factor) for w in ret]
            return ret
class SpladeRewriter(MLMBaseRewriter):
        def logits2weights(self, word_logits):
            ret = [(w[0], np.exp(w[1])) for w in word_logits]
            norm_factor = sum(w[1] for w in ret)
            ret = [(w[0], w[1]/norm_factor) for w in ret]
            return ret