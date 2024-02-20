# Define a lexicon containing words and their POS tags
lexicon = {
    "run": "VB",
    "running": "VBG",
    "ran": "VBD",
    "runs": "VBZ",
    "watched": "VBD",
    "watch": "VB",
    "watches": "VBZ",
    "love": "VB",
    "loves": "VBZ",
    "ice": "NN",
    "cream": "NN",
    "could": "MD",
    "sing": "VB",
    "very": "RB",
    "well": "RB",
    "cat": "NN",
    "toy": "NN",
    "missing": "VBG",
    "have": "VB",
    "two": "CD",
    "dogs": "NNS",
    "they": "PRP",
    "are": "VBP",
    "in": "IN",
    "the": "DT",
    "park": "NN",
    "a": "DT",
    "yesterday": "NN"
}

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs."
]

# Function to tag a sentence using the lexicon
def lexical_tagger(sentence):
    tokens = sentence.split()
    pos_tags = [(token, lexicon.get(token.lower(), "NN")) for token in tokens]
    return pos_tags

# Tag each example sentence using the lexical tagger
for sentence in sentences:
    pos_tags = lexical_tagger(sentence)
    
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(pos_tags)
    print()
