# -*- coding: utf-8 -*-
"""Module for Tibetan stopwords.

This file centralizes the Tibetan stopwords list used in the Tibetan Text Metrics application.
Sources for these stopwords are acknowledged in the main README.md of the webapp.
"""

# Initial set of stopwords with clear categories
PARTICLES_INITIAL = [
    "ཏུ", "གི", "ཀྱི", "གིས", "ཀྱིས", "ཡིས", "ཀྱང", "སྟེ", "ཏེ", "ནོ", "ཏོ",
    "ཅིང", "ཅིག", "ཅེས", "ཞེས", "གྱིས", "ན",
]

MARKERS_AND_PUNCTUATION = ["༈", "།", "༎", "༑"]

ORDINAL_NUMBERS = [
    "དང་པོ", "གཉིས་པ", "གསུམ་པ", "བཞི་པ", "ལྔ་པ", "དྲུག་པ", "བདུན་པ", "བརྒྱད་པ", "དགུ་པ", "བཅུ་པ",
    "བཅུ་གཅིག་པ", "བཅུ་གཉིས་པ", "བཅུ་གསུམ་པ", "བཅུ་བཞི་པ", "བཅོ་ལྔ་པ",
    "བཅུ་དྲུག་པ", "བཅུ་བདུན་པ", "བཅོ་བརྒྱད་པ", "བཅུ་དགུ་པ", "ཉི་ཤུ་པ",
]

# Additional stopwords from the comprehensive list, categorized for readability
MORE_PARTICLES_SUFFIXES = [
    "འི་", "དུ་", "གིས་", "ཏེ", "གི་", "ཡི་", "ཀྱི་", "པས་", "ཀྱིས་", "ཡི", "ལ", "ནི་", "ར", "དུ", 
    "ལས", "གྱིས་", "ས", "ཏེ་", "གྱི་", "དེ", "ཀ་", "སྟེ", "སྟེ་", "ངམ", "ཏོ", "དོ", "དམ་", 
    "གྱིན་", "ན", "འམ་", "ཀྱིན་", "ལོ", "ཀྱིས", "བས་", "ཤིག", "གིས", "ཀི་", "ཡིས་", "གྱི", "གི", 
    "བམ་", "ཤིག་", "ནམ", "མིན་", "ནམ་", "ངམ་", "རུ་", "ཤས་", "ཏུ", "ཡིས", "གིན་", "གམ་", 
    "གྱིས", "ཅང་", "སམ་", "ཞིག", "འང", "རུ", "དང", "ཡ", "འག", "སམ", "ཀ", "འམ", "མམ", 
    "དམ", "ཀྱི", "ལམ", "ནོ་", "སོ་", "རམ་", "བོ་", "ཨང་", "ཕྱི", "ཏོ་", "གེ", "གོ", "རོ་", "བོ", 
    "པས", "འི", "རམ", "བས", "མཾ་", "པོ", "ག་", "ག", "གམ", "བམ", "མོ་", "མམ་", "ཏམ་", "ངོ", 
    "ཏམ", "གིང་", "ཀྱང" # ཀྱང also in PARTICLES_INITIAL, set() will handle duplicates
]

PRONOUNS_DEMONSTRATIVES = ["འདི", "གཞན་", "དེ་", "རང་", "སུ་"]

VERBS_AUXILIARIES = ["ཡིན་", "མི་", "ལགས་པ", "ཡིན་པ", "ལགས་", "མིན་", "ཡིན་པ་", "མིན", "ཡིན་བ", "ཡིན་ལུགས་"]

ADVERBS_QUALIFIERS_INTENSIFIERS = [
    "སོགས་", "ཙམ་", "ཡང་", "ཉིད་", "ཞིང་", "རུང་", "ན་རེ", "འང་", "ཁོ་ན་", "འཕྲལ་", "བར་", 
    "ཅུང་ཟད་", "ཙམ་པ་", "ཤ་སྟག་"
]

QUANTIFIERS_DETERMINERS_COLLECTIVES = [
    "རྣམས་", "ཀུན་", "སྙེད་", "བཅས་", "ཡོངས་", "མཐའ་དག་", "དག་", "ཚུ", "ཚང་མ", "ཐམས་ཅད་", 
    "ཅིག་", "སྣ་ཚོགས་", "སྙེད་པ", "རེ་རེ་", "འགའ་", "སྤྱི", "དུ་མ", "མ", "ཁོ་ན", "ཚོ", "ལ་ལ་", 
    "སྙེད་པ་", "འབའ་", "སྙེད", "གྲང་", "ཁ་", "ངེ་", "ཅོག་", "རིལ་", "ཉུང་ཤས་", "ཚ་"
]

CONNECTORS_CONJUNCTIONS = ["དང་", "ཅིང་", "ཤིང་"]

INTERJECTIONS_EXCLAMATIONS = ["ཨེ་", "འོ་"]

# Combine all categorized lists
_ALL_STOPWORDS_CATEGORIZED = (
    PARTICLES_INITIAL +
    MARKERS_AND_PUNCTUATION +
    ORDINAL_NUMBERS +
    MORE_PARTICLES_SUFFIXES +
    PRONOUNS_DEMONSTRATIVES +
    VERBS_AUXILIARIES +
    ADVERBS_QUALIFIERS_INTENSIFIERS +
    QUANTIFIERS_DETERMINERS_COLLECTIVES +
    CONNECTORS_CONJUNCTIONS +
    INTERJECTIONS_EXCLAMATIONS
)

# Final flat list of unique stopwords
TIBETAN_STOPWORDS = list(set(_ALL_STOPWORDS_CATEGORIZED))

# Final set of unique stopwords for efficient Jaccard/LCS filtering (as a set)
TIBETAN_STOPWORDS_SET = set(TIBETAN_STOPWORDS)
