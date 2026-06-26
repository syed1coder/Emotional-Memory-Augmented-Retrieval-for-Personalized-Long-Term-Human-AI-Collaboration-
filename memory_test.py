import json
from transformers import pipeline

print("Loading emotion model...")

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

print("Model loaded!")

text = "I am really stressed about my thesis deadline."

# Detect emotion
result = classifier(text)

emotion = result[0][0]["label"]

print("\nDetected Emotion:", emotion)

# Create memory entry
memory_entry = {
    "query": text,
    "emotion": emotion
}

# Load existing memories
try:
    with open("emotional_memory.json", "r") as file:
        memory = json.load(file)
except:
    memory = []

# ===== NEW CODE STARTS HERE =====
print("\nRelated memories found:")

found = False

for m in memory:
    if m["emotion"] == emotion:
        print("-", m["query"])
        found = True

if not found:
    print("No related memories found.")
# ===== NEW CODE ENDS HERE =====

# Save new memory
memory.append(memory_entry)

with open("emotional_memory.json", "w") as file:
    json.dump(memory, file, indent=4)

print("\nMemory saved successfully!")