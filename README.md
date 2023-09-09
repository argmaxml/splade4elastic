# Splade4Elastic

A simple query expansion wrapper for elastic search, that uses keyword custom weights derived from an HuggingFace masked-language-model transformer.

## Example usage

    from splade4elastic import SpladeRewriter
    model_name = "roberta-base"
    splader = SpladeRewriter(model_name)
    test_texts = [
        "My name is John",
        "The quick brown fox jumps over the lazy dog",
        "I like to eat apples",
    ]
    for test_text in test_texts:
        print(test_text)
        print(splader.query_expand(test_text))

