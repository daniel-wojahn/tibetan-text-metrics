# Understanding Jaccard Similarity vs. Normalized LCS

When analyzing text similarity, you might encounter situations where the Normalized Longest Common Subsequence (LCS) score is higher than the Jaccard Similarity score. This is not uncommon and provides specific insights into *how* two texts are similar.

## Jaccard Similarity (%)

Jaccard Similarity focuses on the **lexical overlap** between two texts by comparing their sets of *unique* words.

*   **Calculation**: It's calculated as `(Number of common unique words) / (Total number of unique words in both texts combined) * 100`.
*   **Sensitivity**: Jaccard Similarity is insensitive to word order and word frequency. It only considers whether a unique word is present or absent in the sets being compared.
*   **Interpretation**: A higher Jaccard percentage indicates a greater overlap in the vocabularies used.

## Normalized Longest Common Subsequence (LCS)

Normalized LCS measures the length of the longest sequence of words that appears in *both* text segments while maintaining their original relative order.

*   **Key Feature**: The words in the common subsequence do **not** need to be directly adjacent (contiguous) in either of the original texts.
    -   For example, if Text A is **བླ་མ་དེས་ཆོས**་ཟབ་མོ་ཞིག་གསུངས།  and Text B is **བླ་མ་**རིན་པོ་ཆེས་དད་ལྡན་རྣམས་ལ་**ཆོས་**ཀྱི་གདམས་པ་**ཟབ་མོ་ཞིག་གསུངས།** , a possible LCS could be **བླ་མ་ ཆོས་ གསུངས**།. The words "དེས་" and "ཞིག་" from Text A, and "རིན་པོ་ཆེས་དད་ལྡན་རྣམས་ལ་" and "ཀྱི་གདམས་པ་" from Text B are not part of this specific common subsequence.
*   **Normalization**: The length of this common subsequence is then normalized (typically by dividing by the length of the longer of the two segments) to provide a comparable score, often presented as a percentage.
*   **Interpretation**: A higher Normalized LCS score suggests more significant shared phrasing, direct textual borrowing, or strong structural parallelism. It reflects similarities in how ideas are ordered and expressed sequentially.

## When Normalized LCS is higher yhan Jaccard Similarity...

This scenario typically arises when:

1.  **Texts share a strong structural or sequential backbone**: If the texts maintain a similar order of presenting information or share significant phrases and sentence structures, the LCS will capture this ordered similarity effectively.
2.  **Texts also introduce unique vocabulary**: Even with a shared structure, if both texts introduce a considerable number of unique words *around* or *within* that shared structure (e.g., different qualifiers, descriptive terms, or additional clauses not present in the other), these unique words will increase the denominator for the Jaccard calculation (the total unique words), thus lowering the Jaccard score. The LCS is less penalized by these additions as long as the ordered common sequence remains intact.

**Consider this conceptual Tibetan example, reminiscent of offering verses:**

*   **Text A**: "༄༅། **བདག་གིས་མཆོད་པ་**དངོས་སུ་བཤམས་པ་འདི་དག་**འབུལ**།" 
*   **Text B**: "༄༅། **བདག་གིས་མཆོད་པ་**ཡིད་ཀྱིས་སྤྲུལ་པ་རྣམས་ཀྱང་**འབུལ**།" 

*   **LCS**: The phrases "**བདག་གིས་མཆོད་པ་**" and "**འབུལ**།"  form a common subsequence, appearing in the same order. This would lead to a relatively high LCS score.
*   **Jaccard**: The terms "དངོས་སུ་བཤམས་པ་འདི་དག་" in Text A, and "ཡིད་ཀྱིས་སྤྲུལ་པ་རྣམས་ཀྱང་" in Text B are distinct descriptions of the offerings. These differing descriptive phrases increase the total unique word count for the Jaccard denominator, potentially making the Jaccard score lower than the LCS, despite the clear similarity in the core act of offering by the same agent.

In essence, when Normalized LCS is higher than Jaccard Similarity, it often indicates that the similarity between the texts is more strongly characterized by **shared phrasing and order (structural similarity)** than by a simple overlap of their entire unique vocabularies **(lexical similarity)**. The texts might "say similar things in a similar order" for large parts, even if they embellish or vary the surrounding vocabulary.
