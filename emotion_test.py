from transformers import pipeline

print("Loading emotion model...")

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

print("Model loaded!")

text = "I am really stressed about my thesis deadline."

result = classifier(text)

print("\nText:")
print(text)

print("\nDetected Emotion:")
print(result)