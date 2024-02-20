import nltk
from nltk.tag import DefaultTagger, UnigramTagger, HiddenMarkovModelTagger
from nltk.tokenize import word_tokenize

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs."
]

# Rule-based tagging using a default tagger
default_tagger = DefaultTagger('NN')

# Train the Hidden Markov Model (HMM) tagger on the Treebank corpus
train_data = nltk.corpus.treebank.tagged_sents()[:3000]  # Use first 3000 sentences for training
hmm_tagger = HiddenMarkovModelTagger.train(train_data)

# Hybrid tagging function combining rule-based and HMM tagging
def hybrid_tagger(sentence):
    tokens = word_tokenize(sentence)
    # Rule-based tagging
    tagged_sentence = default_tagger.tag(tokens)
    # Update with HMM tagging for unknown words
    for i, (word, tag) in enumerate(tagged_sentence):
        if tag == 'NN':
            tagged_sentence[i] = hmm_tagger.tag([word])[0]
    return tagged_sentence

# Tag each example sentence using the hybrid tagger
for sentence in sentences:
    tagged_sentence = hybrid_tagger(sentence)
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(tagged_sentence)
    print()
