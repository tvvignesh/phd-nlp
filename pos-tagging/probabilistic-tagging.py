import nltk

# Load the Treebank corpus for training
corpus = nltk.corpus.treebank.tagged_sents()

# Split the corpus into train and test sets
train_data = corpus[:3000]  # Use first 3000 sentences for training
test_data = corpus[3000:]   # Use remaining sentences for testing

# Train the Hidden Markov Model (HMM) tagger
hmm_tagger = nltk.HiddenMarkovModelTagger.train(train_data)

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs."
]

# Tag each example sentence using the HMM tagger
for sentence in sentences:
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Tag the tokens using the HMM tagger
    pos_tags = hmm_tagger.tag(tokens)
    
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(pos_tags)
    print()

# Evaluate the HMM tagger on the test data
print("Evaluation on Test Data:")
accuracy = hmm_tagger.accuracy(test_data)
print(f"Accuracy: {accuracy:.2%}")