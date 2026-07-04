import whisper

model = whisper.load_model("base")

def speech_to_text(audio_file):
    result = model.transcribe(audio_file)
    return result["text"].lower()

def introduction_score(text):

    score = 0

    words = len(text.split())

    if words >= 20:
        score += 4

    if "i am" in text or "my age" in text:
        score += 2

    if "live" in text or "from" in text:
        score += 2

    if "student" in text or "work" in text or "job" in text:
        score += 2

    return min(score, 10)

def recall_score(text):

    target_words = [
        "apple",
        "chair",
        "river",
        "school",
        "garden"
    ]

    correct = 0

    for word in target_words:
        if word in text:
            correct += 1

    return correct * 2

def speech_remark(score):

    if score >= 8:
        return "Excellent Cognitive Performance"

    elif score >= 6:
        return "Good Cognitive Performance"

    elif score >= 4:
        return "Moderate Cognitive Concerns"

    else:
        return "Significant Cognitive Concerns"