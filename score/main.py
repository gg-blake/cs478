import ratemyprofessor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ProfessorAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def get_professor(self, name, school_name):
        school = ratemyprofessor.get_school_by_name(school_name)
        professors = ratemyprofessor.get_professors_by_school_and_name(school, name)
        
        for prof in professors:
            if prof.name == name:
                return prof
        return None
    
    def analyze_comments(self, professor):
        ratings = professor.get_ratings()
        weighted_compound = 0
        total_weight = 0
        
        for rating in ratings:
            # Base weight of 1, increased by positive feedback
            weight = 1
            if rating.thumbs_up > 0 or rating.thumbs_down > 0:
                weight += (rating.thumbs_up - rating.thumbs_down) / max(
                    sum(r.thumbs_up + r.thumbs_down for r in ratings), 1)
            
            sentiment = self.sentiment_analyzer.polarity_scores(rating.comment)
            weighted_compound += sentiment['compound'] * weight
            total_weight += weight
        
        avg_sentiment = weighted_compound / total_weight if total_weight > 0 else 0
        
        return {
            'compound': avg_sentiment,
            'pos': max(0, avg_sentiment),
            'neg': max(0, -avg_sentiment),
            'neu': 1 - abs(avg_sentiment)
        }
    
    def calculate_overall_score(self, professor, sentiment):
        weights = {
            'rating': 0.25,
            'sentiment': 0.25,
            'difficulty_inverse': 0.2,
            'would_take_again': 0.3
        }
        
        normalized_scores = {
            'rating': professor.rating / 5.0,
            'sentiment': (sentiment['compound'] + 1) / 2,
            'difficulty_inverse': (5.0 - professor.difficulty) / 5.0,
            'would_take_again': professor.would_take_again / 100 if professor.would_take_again else 0.5
        }
        
        overall_score = sum(weights[key] * normalized_scores[key] for key in weights)
        return overall_score * 100

def main():
    analyzer = ProfessorAnalyzer()
    
    # Example: Daniel Haehn, University of Massachusetts - Boston
    name = input("Enter professor's name: ")
    school = input("Enter school name: ")
    
    professor = analyzer.get_professor(name, school)
    
    if professor is None:
        print("Professor not found.")
        return
    
    print(f"\nProfessor: {professor.name}")
    print(f"Department: {professor.department}")
    print(f"School: {professor.school.name}")
    print(f"Rating: {professor.rating}/5.0")
    print(f"Difficulty: {professor.difficulty}/5.0")
    print(f"Total Ratings: {professor.num_ratings}")
    if professor.would_take_again:
        print(f"Would Take Again: {round(professor.would_take_again, 1)}%")
    else:
        print("Would Take Again: N/A")
    
    sentiment = analyzer.analyze_comments(professor)
    print("\nWeighted Comment Sentiment Analysis:")
    print(f"Positive: {sentiment['pos']:.2f}")
    print(f"Neutral: {sentiment['neu']:.2f}")
    print(f"Negative: {sentiment['neg']:.2f}")
    print(f"Compound: {sentiment['compound']:.2f}")
    
    overall_score = analyzer.calculate_overall_score(professor, sentiment)
    print(f"\nOverall Professor Score: {overall_score:.1f}%")

if __name__ == "__main__":
    main()