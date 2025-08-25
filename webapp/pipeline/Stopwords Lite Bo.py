# -*- coding: utf-8 -*-
"""Module for reduced Tibetan stopwords.

This file provides a less aggressive list of Tibetan stopwords for use in the Tibetan Text Metrics application.
It contains only the most common particles and punctuation that are unlikely to carry significant meaning.
"""

# Initial set of stopwords with clear categories
PARTICLES_INITIAL_LITE = [
    "ཏུ", "གི", "ཀྱི", "གིས", "ཀྱིས", "ཡིས", "ཀྱང", "སྟེ", "ཏེ", "ནོ", "ཏོ",
    "ཅིང", "ཅིག", "ཅེས", "ཞེས", "གྱིས", "ན",
]

MARKERS_AND_PUNCTUATION = ["༈", "།", "༎", "༑"]

# Reduced list of particles and suffixes
MORE_PARTICLES_SUFFIXES_LITE = [
    "འི་", "དུ་", "གིས་", "ཏེ", "གི་", "ཡི་", "ཀྱི་", "པས་", "ཀྱིས་", "ཡི", "ལ", "ནི་", "ར", "དུ", 
    "ལས", "གྱིས་", "ས", "ཏེ་", "གྱི་", "དེ", "ཀ་", "སྟེ", "སྟེ་", "ངམ", "ཏོ", "དོ", "དམ་", 
    "ན", "འམ་", "ལོ", "ཀྱིས", "བས་", "ཤིག", "གིས", "ཀི་", "ཡིས་", "གྱི", "གི"
]

# Combine all categorized lists
_ALL_STOPWORDS_CATEGORIZED_LITE = (
    PARTICLES_INITIAL_LITE +
    MARKERS_AND_PUNCTUATION +
    MORE_PARTICLES_SUFFIXES_LITE
)

# Final flat list of unique stopwords
TIBETAN_STOPWORDS_LITE = list(set(_ALL_STOPWORDS_CATEGORIZED_LITE))

# Final set of unique stopwords for efficient Jaccard/LCS filtering (as a set)
TIBETAN_STOPWORDS_LITE_SET = set(TIBETAN_STOPWORDS_LITE)
