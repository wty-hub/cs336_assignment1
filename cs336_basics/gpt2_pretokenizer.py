"""
Implements the GPT-2 style pre-tokenizer that operates on bytes.
"""

import regex

# The regex pattern used by GPT-2 for pre-tokenization, adapted for bytes.
# This pattern is designed to handle various text structures, including contractions,
# words, numbers, punctuation, and different types of whitespace.
# The pattern is now a bytes pattern (prefixed with b) to operate on UTF-8 encoded text.
#
# Breakdown of the pattern:
# - 's|'t|'re|'ve|'m|'ll|'d: Matches common English contractions.
# -  ?\p{L}+: Matches optional leading space followed by one or more Unicode letters.
# -  ?\p{N}+: Matches optional leading space followed by one or more Unicode numbers.
# -  ?[^\s\p{L}\p{N}]+: Matches optional leading space followed by one or more characters
#    that are NOT whitespace, letters, or numbers (i.e., punctuation).
# - \s+(?!\S): Matches one or more whitespace characters that are not followed by a
#   non-whitespace character. This effectively handles trailing whitespace.
# - \s+: Matches one or more whitespace characters.
GPT2_PRETOKENIZATION_REGEX = regex.compile(
    rb"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class GPT2PreTokenizer:
    """
    Implements the GPT-2 style pre-tokenizer that operates on bytes.

    This pre-tokenizer first encodes a string into UTF-8 bytes, then splits the
    byte sequence into smaller chunks based on a complex regular expression.
    This is the first step in a byte-level BPE (Byte Pair Encoding) tokenization process.
    """

    def __init__(self, special_split_token: bytes):
        """
        Initializes the GPT2PreTokenizer with the compiled bytes regex pattern.
        """
        self.pattern = GPT2_PRETOKENIZATION_REGEX
        self.special_split_token = special_split_token

    def pre_tokenize(self, text: str) -> list[bytes]:
        """
        Encodes the input text to UTF-8 and splits it into a list of byte chunks.

        Args:
            text: The input string to pre-tokenize.

        Returns:
            A list of bytes, where each element is a pre-token chunk.
        """
        # First, encode the string into bytes using UTF-8.
        encoded_text = text.encode("utf-8")
        encoded_text = encoded_text.replace(self.special_split_token, b"")
        # The regex.findall() function finds all non-overlapping matches of the
        # bytes pattern in the encoded text and returns them as a list of bytes.
        return self.pattern.findall(encoded_text)

    def __call__(self, text: str) -> list[bytes]:
        """
        Allows the class instance to be called as a function.
        """
        return self.pre_tokenize(text)
