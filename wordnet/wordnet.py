import nltk
from nltk.corpus import wordnet

# Download WordNet data
nltk.download('wordnet')

# Example word
word = "dog"

# Get synsets for the word "dog"
synsets = wordnet.synsets(word)

# Print information for each synset
for synset in synsets:
    print(f"Synset: {synset.name()}")
    print(f"Definition: {synset.definition()}")
    print(f"Examples: {synset.examples()}")
    
    # Hypernyms: More general concepts (e.g., "animal" is a hypernym of "dog")
    print(f"Hypernyms: {synset.hypernyms()}")
    
    # Hyponyms: More specific concepts (e.g., "poodle" is a hyponym of "dog")
    print(f"Hyponyms: {synset.hyponyms()}")
    
    # Holonyms: Part of the word (e.g., "dog" is a member holonym of "pack")
    print(f"Member Holonyms: {synset.member_holonyms()}")
    
    # Meronyms: Components of the word (e.g., "tail" is a member meronym of "dog")
    print(f"Member Meronyms: {synset.member_meronyms()}")
    
    print()