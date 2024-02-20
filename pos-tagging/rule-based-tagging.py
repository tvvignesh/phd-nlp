import nltk

# Define rules for POS tagging
patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # past tense verbs
    (r'.*es$', 'VBZ'),                # present tense verbs
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                     # nouns (default)
]

# Create a rule-based tagger
rule_based_tagger = nltk.RegexpTagger(patterns)

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs.",
    "12345",
    "This is a default example."
]

# Apply the rule-based tagger to each example sentence
for sentence in sentences:
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Tag the tokens using the rule-based tagger
    pos_tags = rule_based_tagger.tag(tokens)
    
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(pos_tags)
    print()