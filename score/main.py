import ratemyprofessor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Optional

class ProfessorAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def get_professor(self, name: str, school_name: str) -> Optional[ratemyprofessor.Professor]:
        school = ratemyprofessor.get_school_by_name(school_name)
        professors = ratemyprofessor.get_professors_by_school_and_name(school, name)
        
        for prof in professors:
            if prof.name == name:
                return prof
        return None
    
    def analyze_comments(self, professor: ratemyprofessor.Professor) -> Dict[str, float]:
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
    
    def calculate_overall_score(self, professor: ratemyprofessor.Professor, sentiment: Dict[str, float]) -> float:
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

    def analyze_professor(self, name: str, school_name: str) -> Optional[Dict]:
        professor = self.get_professor(name, school_name)
        
        if professor is None:
            print(f"Professor {name} not found.")
            return None
            
        sentiment = self.analyze_comments(professor)
        overall_score = self.calculate_overall_score(professor, sentiment)
        
        return {
            'professor': professor,
            'sentiment': sentiment,
            'overall_score': overall_score
        }

def print_professor_details(analysis: Dict) -> None:
    professor = analysis['professor']
    sentiment = analysis['sentiment']
    overall_score = analysis['overall_score']
    
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
    
    print("\nWeighted Comment Sentiment Analysis:")
    print(f"Positive: {sentiment['pos']:.2f}")
    print(f"Neutral: {sentiment['neu']:.2f}")
    print(f"Negative: {sentiment['neg']:.2f}")
    print(f"Compound: {sentiment['compound']:.2f}")
    
    print(f"\nOverall Professor Score: {overall_score:.1f}%")

def main():
    analyzer = ProfessorAnalyzer()
    
    # Get input for school and professors
    school = input("Enter school name: ")
    print("Enter professor names (one per line, press Enter twice when done):")
    
    professor_names = []
    while True:
        name = input()
        if not name:
            break
        professor_names.append(name)
    
    # Analyze all professors
    professor_analyses = []
    for name in professor_names:
        analysis = analyzer.analyze_professor(name, school)
        if analysis:
            professor_analyses.append(analysis)
    
    # Sort professors by overall score
    professor_analyses.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Print individual details
    print("\n=== Detailed Professor Analyses ===")
    for analysis in professor_analyses:
        print_professor_details(analysis)
        print("\n" + "="*40)
    
    # Print rankings
    print("\n=== Top 3 Professors ===")
    for i, analysis in enumerate(professor_analyses[:3], 1):
        prof = analysis['professor']
        score = analysis['overall_score']
        print(f"{i}. {prof.name} ({prof.department}) - Score: {score:.1f}%")

if __name__ == "__main__":
    main()