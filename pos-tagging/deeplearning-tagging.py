import nltk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the Treebank corpus for training
train_data = nltk.corpus.treebank.tagged_sents()[:3000]  # Use first 3000 sentences for training

# Convert tagged sentences to sequences of word indices and POS tag indices
word_to_index = {word: idx + 1 for idx, word in enumerate(nltk.FreqDist(word.lower() for word, tag in train_data).keys())}
tag_to_index = {tag: idx for idx, tag in enumerate(set(tag for word, tag in train_data))}
max_len = max(len(sentence) for sentence in train_data)

X_train = [[word_to_index.get(word.lower(), 0) for word, tag in sentence] for sentence in train_data]
y_train = [[tag_to_index[tag] for word, tag in sentence] for sentence in train_data]

# Pad sequences to have uniform length
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
y_train = pad_sequences(y_train, maxlen=max_len, padding='post')

# Define the neural network model
model = Sequential([
    Embedding(input_dim=len(word_to_index) + 1, output_dim=50, input_length=max_len),
    Flatten(),
    Dense(units=len(tag_to_index), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs."
]

# Convert example sentences to sequences of word indices
X_test = [[word_to_index.get(word.lower(), 0) for word in nltk.word_tokenize(sentence)] for sentence in sentences]
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Predict POS tags for the example sentences
predicted_tags = model.predict(X_test)
predicted_tags = np.argmax(predicted_tags, axis=-1)

# Map tag indices back to tag names
index_to_tag = {idx: tag for tag, idx in tag_to_index.items()}
predicted_tags = [[index_to_tag[idx] for idx in tags] for tags in predicted_tags]

# Print the example sentences along with predicted POS tags
for sentence, tags in zip(sentences, predicted_tags):
    print(f"Sentence: {sentence}")
    print("Predicted POS Tags:")
    print(tags)
    print()
