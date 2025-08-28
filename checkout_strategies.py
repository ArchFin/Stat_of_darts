# filepath: /Users/a_fin/Documents/Stat_of_darts/checkout_strategies.py
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter

class CheckoutStrategies:
    def __init__(self):
        # Load checkout data if available
        self.checkout_data = None
        self.load_checkout_data()
        
        # Common finishing doubles and their setup scores
        self.finishing_doubles = {
            2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8, 18: 9, 20: 10,
            22: 11, 24: 12, 26: 13, 28: 14, 30: 15, 32: 16, 34: 17, 36: 18, 38: 19, 40: 20,
            50: 25  # Bull
        }
        
    def load_checkout_data(self):
        """Load checkout data from previously generated analysis"""
        try:
            with open('/Users/a_fin/Documents/Stat_of_darts/checkout_probabilities_uniform.json', 'r') as f:
                self.checkout_data = json.load(f)
                print("✅ Loaded existing checkout data")
        except FileNotFoundError:
            print("⚠️ No existing checkout data found. Run checkout_analysis.py first.")
    
    def analyze_checkout_routes(self):
        """Analyze the most common checkout routes and patterns"""
        if not self.checkout_data:
            print("❌ No checkout data available")
            return
        
        print("\n🛣️ Checkout Route Analysis")
        print("=" * 40)
        
        combinations = self.checkout_data['checkout_combinations']
        
        # Analyze finishing double preferences
        finishing_double_counts = Counter()
        total_checkouts = 0
        
        for score, ways in combinations.items():
            score_int = int(score)
            
            # Count 1-dart finishes
            for combo in ways['1_dart']:
                finishing_double_counts[combo[0]] += 1
                total_checkouts += 1
            
            # Count 2-dart finishes
            for combo in ways['2_dart']:
                finishing_double_counts[combo[1]] += 1
                total_checkouts += 1
            
            # Count 3-dart finishes
            for combo in ways['3_dart']:
                finishing_double_counts[combo[2]] += 1
                total_checkouts += 1
        
        print(f"\n🎯 Most Popular Finishing Doubles:")
        most_common_finishes = finishing_double_counts.most_common(10)
        for i, (finish, count) in enumerate(most_common_finishes):
            percentage = (count / total_checkouts) * 100
            print(f"   {i+1:2d}. Double {finish//2 if finish != 50 else 'Bull'} ({finish}): {count:4d} ways ({percentage:.1f}%)")
        
        # Analyze setup patterns
        self.analyze_setup_patterns(combinations)
        
        return finishing_double_counts
    
    def analyze_setup_patterns(self, combinations):
        """Analyze common setup patterns for checkouts"""
        print(f"\n🎲 Setup Pattern Analysis:")
        
        # Analyze 2-dart checkout setups
        setup_scores = Counter()
        
        for score, ways in combinations.items():
            score_int = int(score)
            
            # Analyze first dart in 2-dart checkouts
            for combo in ways['2_dart']:
                setup_scores[combo[0]] += 1
        
        print(f"\n   Most Common Setup Scores (2-dart checkouts):")
        for i, (setup, count) in enumerate(setup_scores.most_common(10)):
            print(f"   {i+1:2d}. Score {setup}: {count} times")
        
        # Analyze impossible checkout scenarios
        self.analyze_impossible_checkouts()
    
    def analyze_impossible_checkouts(self):
        """Analyze scores that cannot be checked out"""
        impossible_scores = []
        
        # Check which scores from 2-170 are impossible
        for score in range(2, 171):
            if self.checkout_data and str(score) in self.checkout_data['probabilities']:
                prob_data = self.checkout_data['probabilities'][str(score)]
                if prob_data['total'] == 0:
                    impossible_scores.append(score)
        
        print(f"\n❌ Impossible Checkout Scores:")
        print(f"   Scores that cannot be finished: {impossible_scores}")
        
        # Analyze why they're impossible
        print(f"\n🔍 Why These Scores Are Impossible:")
        for score in impossible_scores:
            if score == 169:
                print(f"   {score}: Odd number > 50, cannot finish on double")
            elif score == 168:
                print(f"   {score}: Would need triple 20 + double 54 (doesn't exist)")
            elif score == 166:
                print(f"   {score}: Would need triple 20 + double 53 (doesn't exist)")
            elif score == 165:
                print(f"   {score}: Odd number, would need setup leaving odd remainder")
            elif score == 163:
                print(f"   {score}: Odd number, would need setup leaving odd remainder")
            elif score == 162:
                print(f"   {score}: Would need triple 20 + double 51 (doesn't exist)")
            elif score == 159:
                print(f"   {score}: Odd number, would need setup leaving odd remainder")
    
    def compare_checkout_strategies(self):
        """Compare different checkout strategies"""
        print(f"\n⚔️ Checkout Strategy Comparison")
        print("=" * 40)
        
        strategies = {
            'conservative': "Always aim for largest possible double",
            'aggressive': "Go for high-scoring setups even if risky",
            'balanced': "Mix of safety and scoring"
        }
        
        print(f"\nStrategy Approaches:")
        for strategy, description in strategies.items():
            print(f"   {strategy.title()}: {description}")
        
        # Analyze specific examples
        example_scores = [100, 87, 76, 41, 32]
        
        print(f"\n📋 Example Checkout Decisions:")
        for score in example_scores:
            if self.checkout_data and str(score) in self.checkout_data['checkout_combinations']:
                ways = self.checkout_data['checkout_combinations'][str(score)]
                print(f"\n   Score {score}:")
                
                if ways['1_dart']:
                    print(f"     1-dart: {ways['1_dart'][0]}")
                
                if ways['2_dart']:
                    # Show best 2-dart options
                    best_2_dart = sorted(ways['2_dart'], key=lambda x: x[0], reverse=True)[:3]
                    print(f"     Best 2-dart options:")
                    for combo in best_2_dart:
                        print(f"       {combo[0]} → Double {combo[1]//2 if combo[1] != 50 else 'Bull'}")
                
                if ways['3_dart']:
                    # Show sample 3-dart options
                    sample_3_dart = ways['3_dart'][:3]
                    print(f"     Sample 3-dart options:")
                    for combo in sample_3_dart:
                        finish_name = f"Double {combo[2]//2}" if combo[2] != 50 else "Bull"
                        print(f"       {combo[0]} → {combo[1]} → {finish_name}")
    
    def plot_checkout_difficulty_by_range(self):
        """Plot checkout difficulty by score ranges"""
        if not self.checkout_data:
            print("❌ No checkout data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        scores = []
        total_probs = []
        ways_counts = []
        
        for score_str, prob_data in self.checkout_data['probabilities'].items():
            score = int(score_str)
            scores.append(score)
            total_probs.append(prob_data['total'])
            total_ways = prob_data['ways_1'] + prob_data['ways_2'] + prob_data['ways_3']
            ways_counts.append(total_ways)
        
        # Sort by score
        sorted_data = sorted(zip(scores, total_probs, ways_counts))
        scores, total_probs, ways_counts = zip(*sorted_data)
        
        # Plot 1: Probability by score ranges
        score_ranges = {
            'Easy (2-50)': [s for s in scores if 2 <= s <= 50],
            'Medium (51-100)': [s for s in scores if 51 <= s <= 100],
            'Hard (101-150)': [s for s in scores if 101 <= s <= 150],
            'Extreme (151-170)': [s for s in scores if 151 <= s <= 170]
        }
        
        range_avg_probs = []
        range_names = []
        
        for range_name, range_scores in score_ranges.items():
            if range_scores:
                range_probs = [total_probs[scores.index(s)] for s in range_scores]
                avg_prob = np.mean(range_probs)
                range_avg_probs.append(avg_prob)
                range_names.append(range_name)
        
        bars1 = ax1.bar(range_names, range_avg_probs, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax1.set_ylabel('Average Checkout Probability')
        ax1.set_title('Average Checkout Probability by Score Range')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, prob in zip(bars1, range_avg_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{prob:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Distribution of checkout difficulties
        ax2.hist(total_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Checkout Probability')
        ax2.set_ylabel('Number of Scores')
        ax2.set_title('Distribution of Checkout Difficulties')
        ax2.axvline(np.mean(total_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(total_probs):.4f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/a_fin/Documents/Stat_of_darts/checkout_strategy_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_checkout_cheat_sheet(self):
        """Generate a practical checkout cheat sheet"""
        if not self.checkout_data:
            print("❌ No checkout data available")
            return
        
        print(f"\n📋 Checkout Cheat Sheet")
        print("=" * 50)
        
        combinations = self.checkout_data['checkout_combinations']
        
        # Focus on common checkout scores
        common_scores = [170, 167, 164, 161, 158, 156, 155, 152, 150, 141, 140, 
                        138, 137, 136, 135, 132, 130, 129, 128, 127, 126, 125, 
                        124, 123, 122, 121, 120, 110, 100, 90, 80, 70, 60, 50, 40, 32]
        
        cheat_sheet = {}
        
        for score in common_scores:
            if str(score) in combinations:
                ways = combinations[str(score)]
                best_routes = []
                
                # Prefer 1-dart if possible
                if ways['1_dart']:
                    best_routes.append(f"1-dart: {ways['1_dart'][0]}")
                
                # Best 2-dart routes (prefer higher first dart)
                if ways['2_dart']:
                    best_2_dart = sorted(ways['2_dart'], key=lambda x: x[0], reverse=True)[:2]
                    for combo in best_2_dart:
                        finish_name = f"D{combo[1]//2}" if combo[1] != 50 else "Bull"
                        best_routes.append(f"2-dart: {combo[0]} → {finish_name}")
                
                # Sample 3-dart route
                if ways['3_dart'] and not ways['1_dart'] and not ways['2_dart']:
                    combo = ways['3_dart'][0]
                    finish_name = f"D{combo[2]//2}" if combo[2] != 50 else "Bull"
                    best_routes.append(f"3-dart: {combo[0]} → {combo[1]} → {finish_name}")
                
                cheat_sheet[score] = best_routes[:3]  # Top 3 routes
        
        # Print cheat sheet
        for score in sorted(cheat_sheet.keys(), reverse=True):
            print(f"\n{score:3d}:")
            for route in cheat_sheet[score]:
                print(f"     {route}")
        
        # Save to file
        with open('/Users/a_fin/Documents/Stat_of_darts/checkout_cheat_sheet.txt', 'w') as f:
            f.write("DARTBOARD CHECKOUT CHEAT SHEET\n")
            f.write("=" * 50 + "\n\n")
            
            for score in sorted(cheat_sheet.keys(), reverse=True):
                f.write(f"{score:3d}:\n")
                for route in cheat_sheet[score]:
                    f.write(f"     {route}\n")
                f.write("\n")
        
        print(f"\n💾 Cheat sheet saved to checkout_cheat_sheet.txt")

def main():
    """Main function to run checkout strategy analysis"""
    strategies = CheckoutStrategies()
    
    print("🎯 Dartboard Checkout Strategy Analysis")
    print("=" * 50)
    
    # Analyze checkout routes and patterns
    strategies.analyze_checkout_routes()
    
    # Compare different strategies
    strategies.compare_checkout_strategies()
    
    # Create visualizations
    strategies.plot_checkout_difficulty_by_range()
    
    # Generate practical cheat sheet
    strategies.generate_checkout_cheat_sheet()
    
    print(f"\n✅ Checkout strategy analysis complete!")

if __name__ == "__main__":
    main()