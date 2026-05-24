import os
import pymongo
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class SentimentPipeline:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(os.getenv("MONGODB_URL"))
            self.db = self.client["healthcare_sentiment"]
            self.collection = self.db["results"]
            print("✅ MongoDB Connected!")
        except Exception as e:
            print(f"❌ MongoDB Error: {e}")

    def save_result(self, text: str, result: dict, source: str = "manual"):
        try:
            document = {
                "text": str(text)[:500],
                "sentiment": result["sentiment"],
                "emoji": result["emoji"],
                "polarity": result["polarity"],
                "subjectivity": result.get("subjectivity", 0),
                "confidence": result["confidence"],
                "model": result["model"],
                "source": source,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.collection.insert_one(document)
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def save_batch(self, texts: list, results: list, source: str = "batch"):
        try:
            documents = []
            for text, result in zip(texts, results):
                documents.append({
                    "text": str(text)[:500],
                    "sentiment": result["sentiment"],
                    "emoji": result["emoji"],
                    "polarity": result["polarity"],
                    "subjectivity": result.get("subjectivity", 0),
                    "confidence": result["confidence"],
                    "model": result["model"],
                    "source": source,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            if documents:
                self.collection.insert_many(documents)
            return len(documents)
        except Exception as e:
            print(f"Batch save error: {e}")
            return 0

    def get_history(self, limit: int = 100) -> list:
        try:
            return list(
                self.collection.find(
                    {}, {"_id": 0}
                ).sort("timestamp", -1).limit(limit)
            )
        except:
            return []

    def get_stats(self) -> dict:
        try:
            total = self.collection.count_documents({})
            positive = self.collection.count_documents({"sentiment": "POSITIVE"})
            negative = self.collection.count_documents({"sentiment": "NEGATIVE"})
            neutral = self.collection.count_documents({"sentiment": "NEUTRAL"})
            return {
                "total": total,
                "positive": positive,
                "negative": negative,
                "neutral": neutral
            }
        except:
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0}

    def clear_history(self):
        try:
            self.collection.delete_many({})
            return True
        except:
            return False