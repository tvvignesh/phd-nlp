import nltk
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
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

# Train unigram, bigram, and trigram taggers on the Treebank corpus
train_data = nltk.corpus.treebank.tagged_sents()[:3000]  # Use first 3000 sentences for training
unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)
bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(train_data, backoff=bigram_tagger)

# Ensemble tagging function combining multiple taggers
def ensemble_tagger(sentence):
    tokens = word_tokenize(sentence)
    # Tag sentence using all taggers
    tagged_default = default_tagger.tag(tokens)
    tagged_unigram = unigram_tagger.tag(tokens)
    tagged_bigram = bigram_tagger.tag(tokens)
    tagged_trigram = trigram_tagger.tag(tokens)
    # Combine tags using a voting mechanism
    ensemble_tags = []
    for i in range(len(tokens)):
        tags = [tagged_default[i][1], tagged_unigram[i][1], tagged_bigram[i][1], tagged_trigram[i][1]]
        ensemble_tags.append(max(set(tags), key=tags.count))
    return list(zip(tokens, ensemble_tags))

# Tag each example sentence using the ensemble tagger
for sentence in sentences:
    tagged_sentence = ensemble_tagger(sentence)
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(tagged_sentence)
    print()
