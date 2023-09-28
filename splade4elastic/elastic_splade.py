import itertools, collections, json, string, re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List
import numpy as np


class MLMBaseRewriter:
    """Elastic SPLADE model"""

    def __init__(self, model_name: str, expansions_per_word:int = 10, multi_word="split", exluded_words=[]):
        """Initialize the model

        Args:
            model_name (str): name of the model
            multi_word (str, optional): How to handle multi-word tokens. Defaults to "split". Can be "filter" or "ignore"
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token="<s>")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.k=expansions_per_word
        self.exluded_words = exluded_words
        self.const_weight = 1
        self.multi_word = multi_word
        self.vocab = self.read_vocab() if multi_word == "filter" else None

    def read_vocab(self, vocab='/usr/share/dict/words'):
        try: 
            with open(vocab, 'r') as f:
                words = [l.strip() for l in f.readlines()]
        except FileNotFoundError:
            print(f"Could not find {vocab} file, using empty vocab")
            return set()
        words = [w.lower() for w in words if len(w)>1]
        return frozenset(words)

    def __tokenize_to_words(self, sentence):
        # Split the sentence into words
        words = sentence.split()

        # Define a translation table to replace punctuation marks with spaces
        translation_table = str.maketrans('', '', string.punctuation)

        # Initialize a list to store the cleaned words
        cleaned_words = []

        for word in words:
            # Check if the word is not a special token
            if word not in self.tokenizer.all_special_tokens:
                # Remove punctuation marks from the word
                cleaned_word = word.translate(translation_table)
                cleaned_words.append(cleaned_word)
            else:
                # If the word is a special token, add it as is
                cleaned_words.append(word)

        return cleaned_words
    
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
        return word not in self.exluded_words # expand all words except for the excluded ones
    
    def mask_expansion(self, txt):
        ret = collections.defaultdict(list)
        special_tokens = self.tokenizer.all_special_tokens
        if self.tokenizer.bos_token:
            txt = self.tokenizer.bos_token + ' ' + txt
        X = self.tokenizer.encode(txt, return_tensors="pt")
        word2token = self.__tokenize_and_preserve(txt)
        words = self.__tokenize_to_words(txt)
        for wi, lst in word2token:
            if not self.do_expansion(words[wi]):
                # skip this word
                ret[wi].append((words[wi], self.const_weight))
                continue
            if self.multi_word == "ignore" and len(lst) > 1: # skip multi-word tokens
                ret[wi].append((words[wi], self.const_weight))
                continue
            X_m = X.clone()
            for mask_token_index, token, _ in lst:
                ti = mask_token_index
                # if self.tokenizer.bos_token:
                #     ti += 1
                X_m[0, ti] = self.tokenizer.mask_token_id
            logits = self.model(X_m).logits
            all_combinations = []
            for mask_token_index, token, _ in lst:
                mask_token_logits = logits[0, mask_token_index, :] # need to add 1 because of the bos token we added
                max_ids = np.argsort(mask_token_logits.to("cpu").detach().numpy())[
                    ::-1
                ][:self.k]
                max_tokens = self.tokenizer.convert_ids_to_tokens(max_ids)
                # max_tokens = [t[1:] if t.startswith("Ġ") else t for t in max_tokens] # remove the leading space
                max_scores = np.sort(mask_token_logits.to("cpu").detach().numpy())[::-1][ :self.k]
                tokens_tuple = zip(max_tokens, max_scores)
                sub_words = [(t, s) for t, s in tokens_tuple if t not in special_tokens]
                all_combinations.append(sub_words)

            # create all possible combinations of sub-words and normalize their scores
            all_combinations = self.combine_and_normalize(all_combinations)
            all_combinations = [(w[1:], s) if w.startswith("Ġ") else (w, s) for w, s in all_combinations]

            if self.multi_word == "filter":
                # filter out sub-words that are not in linux built-in dictionary
                all_combinations = [(w, s) for w, s in all_combinations if w.lower() in self.vocab or len(w.split(" ")) > 1]
                
            all_combinations = [(w, s) for w, s in all_combinations if len(w) > 0] # filter out empty sub-words
            ret[wi].extend(all_combinations)
                    
        ret = dict(ret)
        return list(ret.values())

    
    def combine_and_normalize(self, all_combinations):
        result = []
        
        # Filter out empty sub-lists
        non_empty_combinations = [sub_list for sub_list in all_combinations if sub_list]
        
        # Check if there are any non-empty sub-lists
        if not non_empty_combinations:
            return result
        
        # Initialize with the first non-empty sub-list
        initial_combination = non_empty_combinations[0]
        
        # Check if there's only one non-empty sub-list
        if len(non_empty_combinations) == 1:
            return initial_combination
        
        # Create a dictionary to store maximum scores for each word
        max_scores = {word: max(score for _, score in sub_list) for sub_list in non_empty_combinations for word, _ in sub_list}
        
        # Iterate through all possible combinations of sub-words
        for sub_word_combination in itertools.product(*non_empty_combinations):
            combined_sub_words = ''.join(word.replace('Ġ', ' ') for word, _ in sub_word_combination) # Combine the sub-words and use 'Ġ' to decide where to add spaces
            
            # Calculate the product of scores for the sub-words in the combination
            combined_score = 1.0  # Initialize with a score of 1.0
            for word, score in sub_word_combination:
                combined_score *= score / max_scores[word]  # Normalize the score
            
            # Append the combined sub-words and their normalized score
            result.append((combined_sub_words, combined_score))
        
        # Sort the result by normalized scores (optional)
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result

    def __only_alpha(self, txt): 
        # consider to delete this function
        return "".join(c for c in txt if c in string.ascii_letters)
    
    def logits2weights(self, word_logits):
            return word_logits
    
    def __force_weights(self, word_logits, txt):
        ret = [(w[0], self.const_weight) if w[0] in txt else w for w in word_logits]
        return ret
    
    def __elastic_format(self, expanded_list, text):
        ret = []
        text = text.lower().split()
        # print(text)
        for words in expanded_list:
            # print(words)
            words = self.logits2weights(words)
            # print(words)
            # The original word should have a higher score
            words = self.__force_weights(words, text)
            words = [(w[0].lower(), w[1]) for w in words if w[0] != self.tokenizer.bos_token]
            # unite equal words and sum their scores
            unique_words = {w[0] for w in words}
            words = [(unique, sum(w[1] for w in words if w[0] == unique)) for unique in unique_words]
            # sort by score
            words = sorted(words, key=lambda x: x[1], reverse=True)
            or_statement_list = [f"{w[0]}^{round(float(w[1]), 2)}" for w in words]
            or_statement = " OR ".join(or_statement_list)
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
            if len(word_logits) == 0:
                return word_logits # empty list
            min_score = min(w[1] for w in word_logits) 
            ret = [(w[0], w[1] - min_score) for w in word_logits]
            norm_factor = sum(w[1] for w in ret)
            ret = [(w[0], w[1]/norm_factor) for w in ret]
            return ret
class SpladeRewriter(MLMBaseRewriter):
        def logits2weights(self, word_logits):
            if len(word_logits) == 0:
                return word_logits # empty list
            ret = [(w[0], np.exp(w[1])) for w in word_logits]
            norm_factor = sum(w[1] for w in ret)
            ret = [(w[0], w[1]/norm_factor) for w in ret]
            return ret