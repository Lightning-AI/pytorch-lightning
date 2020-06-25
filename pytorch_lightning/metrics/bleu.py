from collections import Counter
import math
import re
from typing import List


def math_log(num: int):
    return -99999999 if num == 0.0 else math.log(num)


def best_match_len(translate_list: List[list], reference_list: List[list]):
    """Finding the closet length of the reference texts with respect to translated text

    Args:
        translate_list: A list of translated text
        reference_list: A list of reference texts

    Return:
        ref_len: the best match length of text of the reference lists with translated text
        translate_len: the length of translated text
    """
    translate_len = len(translate_list)
    len_diff_lists = [abs(len(ref.split()) - translate_len) for ref in reference_list]
    best_match_len_index = len_diff_lists.index(min(len_diff_lists))
    ref_len = len(reference_list[best_match_len_index].split())

    return ref_len, translate_len


def count_ngram(ngram_input_list: List[list], num_grams: int) -> Counter:
    """Counting how many times each word appears in a given text with ngram

    Args:
        ngram_input_list: A list of translated text or reference texts
        num_grams: gram value ranged 1 to 4

    Return:
        ngram_counter: Counter object of ngram
    """
    ngram_counter = Counter()

    for i in range(1, num_grams + 1):
        for j in range(len(ngram_input_list) - i + 1):
            ngram_key = " ".join(ngram_input_list[j : i + j])
            ngram_counter[ngram_key] += 1

    return ngram_counter


def count_clip_ngram(translate_list: List[list], reference_list: List[list], num_grams: int) -> Counter:
    """Clipping the count value of each word to be minimum value from translated counter and reference counter
    Args:
        translate_list: A list of translated text
        reference_list: A list of reference texts
        num_grams: gram value

    Return:
        ngram_counter_clip: Counter object of clipped ngram
    """
    translate_ngram_counter = count_ngram(translate_list, num_grams)
    reference_ngram_counter_list = []
    reference_ngram_counter = Counter()

    for ref in reference_list:
        reference_ngram_counter_list.append(count_ngram(ref.split(), num_grams))

    for ref in reference_ngram_counter_list:
        reference_ngram_counter |= ref

    ngram_counter_clip = translate_ngram_counter & reference_ngram_counter

    return ngram_counter_clip


def bleu_score(translate_text: str, reference_list: List[list], n: int = 4) -> float:
    """Calculate bleu score of machine translated text

    Args:
        translate_text: A string of translated text
        reference_list: A list of reference texts
        n: gram value ranged from 1 to 4 (Default 4)

    Return:
        Bleu Score

    Example:

        >>> t = 'the FAST brown fox jumped over the lazy dog'
        >>> r = ['the quick brown fox jumped over the lazy dog', 'the quick brown fox jumped over the the lazy cat']
        >>> bleu_score(t, r)
        0.7506
    """
    translate_list = translate_text.lower().split()
    if len(translate_list) < n:
        raise ValueError("Translated text should be longer or equal to gram value.")

    for ref in reference_list:
        if len(ref.split()) < n:
            raise ValueError("Reference texts in the reference list should be longer or equal to gram value.")

    reference_list = [ref.lower() for ref in reference_list]
    numerator = [0] * n
    denominator = [0] * n
    precision_scores = [0] * n
    r, c = best_match_len(translate_list, reference_list)

    ngram_counter = count_ngram(translate_list, n)
    ngram_counter_clip = count_clip_ngram(translate_list, reference_list, n)

    for counter_clip in ngram_counter_clip:
        numerator[len(counter_clip.split()) - 1] += ngram_counter[counter_clip]

    for counter in ngram_counter:
        denominator[len(counter.split()) - 1] += ngram_counter[counter]

    for i in range(n):
        precision_scores[i] = numerator[i] / denominator[i]

    geometric_mean = math.exp(sum((1.0 / n) * math_log(p) for p in precision_scores))
    bp = 1.0 if c > r else math.exp(1 - (r / c))
    bleu = round(bp * geometric_mean, 4)

    return bleu


# t = "the FAST brown fox jumped over the lazy dog"
# r = ['this is a small test', 'the quick brown fox jumped over the lazy dog', 'the quick brown fox jumped over the the lazy cat']

# t = 'this is a test'
# r = ['this is a test', 'this is test']

# t = "this is a test"
# r = ["this is small test"]

# t = 'the the the the the the the'
# r = ['the cat is on the mat', 'there is a cat on the mat']

t = "the FAST brown fox jumped over the lazy dog"
r = ["the quick brown fox jumped over the lazy dog", "the quick brown fox jumped over the the lazy cat"]

# t = "pytorch_lightning is awesome"
# r = ["THIS IS AWESOME PYTORCH_LIGHTNING", "AWESOME PYTORCH LIGHTNING"]

print(bleu_score(t, r, 4))
