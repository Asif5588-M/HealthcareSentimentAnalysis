from src.data import DataProcessor
from src.model import SentimentModel
from src.pipeline import SentimentPipeline


def main():
    print("🏥 Healthcare Sentiment Analysis Pipeline!")
    print("=" * 50)

    processor = DataProcessor()
    model = SentimentModel()
    pipeline = SentimentPipeline()

    reviews = processor.get_sample_healthcare_reviews()
    print(f"\n📊 Analyzing {len(reviews)} healthcare reviews...\n")

    results = model.analyze_batch(reviews)

    saved = pipeline.save_batch(reviews, results, source="sample")
    print(f"✅ {saved} results saved to MongoDB!\n")

    for result in results:
        print(
            f"{result['emoji']} {result['sentiment']:8} "
            f"(polarity: {result['polarity']:+.3f}) "
            f"— {result['text'][:60]}..."
        )

    stats = pipeline.get_stats()
    print(f"\n{'=' * 50}")
    print(f"📈 Final Stats:")
    print(f"Total Analyzed : {stats['total']}")
    print(f"😊 Positive    : {stats['positive']}")
    print(f"😞 Negative    : {stats['negative']}")
    print(f"😐 Neutral     : {stats['neutral']}")
    print(f"{'=' * 50}")
    print("\n✅ Pipeline Complete!")


if __name__ == "__main__":
    main()