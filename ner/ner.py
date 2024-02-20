from transformers import pipeline

# Load the named entity recognition (NER) pipeline
ner_pipeline = pipeline("ner", grouped_entities=True)

# Example sentences
sentences = [
    "Barack Obama was the 44th President of the United States.",
    "The Eiffel Tower is located in Paris, France.",
    "Elon Musk is the CEO of SpaceX and Tesla.",
    "Albert Einstein was a German-born theoretical physicist.",
    "The Amazon River is the largest river by discharge volume in the world."
]

# Perform named entity recognition on each example sentence
for sentence in sentences:
    # Perform NER on the sentence
    entities = ner_pipeline(sentence)
    
    # Print the named entities detected in the sentence
    print(f"Sentence: {sentence}")
    print("Named Entities:")
    for entity in entities:
        print(f"Entity: {entity['entity']}, Type: {entity['entity_group']}, Score: {entity['score']}, Start: {entity['start']}, End: {entity['end']}, Text: {entity['word']}")
    print()