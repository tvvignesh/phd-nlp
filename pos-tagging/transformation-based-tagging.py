import nltk
from nltk.tag import brill, brill_trainer
from nltk.tag import UnigramTagger
from nltk.corpus import treebank

# Load the Treebank corpus for training
train_data = treebank.tagged_sents()[:3000]  # Use first 3000 sentences for training

# Define a function to extract features from tagged sentences
def get_features(sentence, index):
    word, pos = sentence[index]
    return {
        'word': word,
        'prev_word': '' if index == 0 else sentence[index-1][0],
        'next_word': '' if index == len(sentence)-1 else sentence[index+1][0],
    }

# Define the Brill tagger trainer
brill_trainer = brill_trainer.BrillTaggerTrainer(initial_tagger=UnigramTagger(train_data), templates=brill.brill24())

# Train the Brill tagger
brill_tagger = brill_trainer.train(train_data, max_rules=10)

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs."
]

# Tag each example sentence using the Brill tagger
for sentence in sentences:
    # Tokenize the sentence and tag it using the Brill tagger
    tokens = nltk.word_tokenize(sentence)
    tagged_sentence = brill_tagger.tag(tokens)
    
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(tagged_sentence)
    print()
