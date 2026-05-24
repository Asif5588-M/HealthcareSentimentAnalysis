from textblob import TextBlob
import nltk

nltk.download('punkt', quiet=True)


class SentimentModel:
    def __init__(self):
        print("✅ Sentiment Model ready!")

    def analyze_textblob(self, text: str) -> dict:
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                sentiment = "POSITIVE"
                emoji = "😊"
                color = "#28a745"
            elif polarity < -0.1:
                sentiment = "NEGATIVE"
                emoji = "😞"
                color = "#dc3545"
            else:
                sentiment = "NEUTRAL"
                emoji = "😐"
                color = "#ffc107"

            return {
                "sentiment": sentiment,
                "emoji": emoji,
                "color": color,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "confidence": round(abs(polarity) * 100, 1),
                "model": "TextBlob"
            }
        except Exception as e:
            raise Exception(f"Analysis error: {e}")

    def analyze_batch(self, texts: list) -> list:
        results = []
        for text in texts:
            result = self.analyze_textblob(text)
            result['text'] = text
            results.append(result)
        return results