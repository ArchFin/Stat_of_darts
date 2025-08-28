# filepath: /Users/a_fin/Documents/Stat_of_darts/checkout_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, Counter
import time
from itertools import combinations_with_replacement, permutations
import json

class CheckoutAnalyzer:
    def __init__(self):
        # Official dartboard layout (clockwise from top)
        self.sectors = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        
        # All possible single dart scores
        self.single_scores = list(range(1, 21))  # 1-20
        self.double_scores = [i * 2 for i in range(1, 21)]  # 2, 4, 6, ..., 40
        self.triple_scores = [i * 3 for i in range(1, 21)]  # 3, 6, 9, ..., 60
        self.bull_scores = [25, 50]  # Outer bull, Inner bull
        
        # All possible dart scores (no miss)
        self.all_dart_scores = sorted(list(set(
            self.single_scores + self.double_scores + self.triple_scores + self.bull_scores
        )))
        
        # Double-out finishing scores (must end on double)
        self.finishing_doubles = self.double_scores + [50]  # Include bull (50) as it counts as double
        
        # Generate all possible checkout combinations
        self.checkout_combinations = self.generate_checkout_combinations()
        
    def generate_checkout_combinations(self):
        """Generate all possible checkout combinations for scores 2-170"""
        print("🎯 Generating checkout combinations...")
        start_time = time.time()
        
        checkout_combinations = {}
        
        # For each possible remaining score
        for remaining_score in range(2, 171):  # 2 is minimum checkout, 170 is maximum
            checkout_combinations[remaining_score] = self.find_checkout_ways(remaining_score)
            
        end_time = time.time()
        print(f"✅ Generated checkout combinations in {end_time - start_time:.2f} seconds")
        
        return checkout_combinations
    
    def find_checkout_ways(self, remaining_score):
        """Find all ways to checkout from a given remaining score"""
        ways = {
            '1_dart': [],
            '2_dart': [],
            '3_dart': []
        }
        
        # 1-dart checkout (must be a finishing double)
        if remaining_score in self.finishing_doubles:
            ways['1_dart'].append([remaining_score])
        
        # 2-dart checkout (second dart must be a finishing double)
        for first_dart in self.all_dart_scores:
            remaining_after_first = remaining_score - first_dart
            if remaining_after_first > 0 and remaining_after_first in self.finishing_doubles:
                ways['2_dart'].append([first_dart, remaining_after_first])
        
        # 3-dart checkout (third dart must be a finishing double)
        for first_dart in self.all_dart_scores:
            remaining_after_first = remaining_score - first_dart
            if remaining_after_first <= 0:
                continue
                
            for second_dart in self.all_dart_scores:
                remaining_after_second = remaining_after_first - second_dart
                if remaining_after_second > 0 and remaining_after_second in self.finishing_doubles:
                    ways['3_dart'].append([first_dart, second_dart, remaining_after_second])
        
        return ways
    
    def calculate_checkout_probabilities(self, skill_model='uniform'):
        """Calculate checkout probabilities for different skill models"""
        print(f"📊 Calculating checkout probabilities (skill model: {skill_model})...")
        
        probabilities = {}
        
        if skill_model == 'uniform':
            # Uniform probability for all dart scores
            single_dart_prob = 1.0 / len(self.all_dart_scores)
            dart_probs = {score: single_dart_prob for score in self.all_dart_scores}
        elif skill_model == 'skilled':
            # Higher probability for higher scoring areas
            dart_probs = self.create_skilled_probabilities()
        else:
            raise ValueError(f"Unknown skill model: {skill_model}")
        
        for remaining_score in range(2, 171):
            ways = self.checkout_combinations[remaining_score]
            
            # Calculate probability for each dart count
            prob_1_dart = 0.0
            prob_2_dart = 0.0
            prob_3_dart = 0.0
            
            # 1-dart checkouts
            for combo in ways['1_dart']:
                prob_1_dart += dart_probs[combo[0]]
            
            # 2-dart checkouts
            for combo in ways['2_dart']:
                prob_2_dart += dart_probs[combo[0]] * dart_probs[combo[1]]
            
            # 3-dart checkouts
            for combo in ways['3_dart']:
                prob_3_dart += dart_probs[combo[0]] * dart_probs[combo[1]] * dart_probs[combo[2]]
            
            # Total checkout probability (any number of darts)
            total_prob = prob_1_dart + prob_2_dart + prob_3_dart
            
            probabilities[remaining_score] = {
                '1_dart': prob_1_dart,
                '2_dart': prob_2_dart,
                '3_dart': prob_3_dart,
                'total': total_prob,
                'ways_1': len(ways['1_dart']),
                'ways_2': len(ways['2_dart']),
                'ways_3': len(ways['3_dart'])
            }
        
        return probabilities
    
    def create_skilled_probabilities(self):
        """Create probability distribution favoring higher scores (skilled player model)"""
        dart_probs = {}
        
        # Base probabilities (higher for higher scores)
        for score in self.all_dart_scores:
            if score >= 50:  # High scores (50, 60)
                dart_probs[score] = 0.02
            elif score >= 40:  # Good scores (40-49)
                dart_probs[score] = 0.025
            elif score >= 20:  # Medium scores (20-39)
                dart_probs[score] = 0.02
            elif score >= 10:  # Low scores (10-19)
                dart_probs[score] = 0.015
            else:  # Very low scores (1-9)
                dart_probs[score] = 0.01
        
        # Normalize to sum to 1
        total_prob = sum(dart_probs.values())
        dart_probs = {score: prob/total_prob for score, prob in dart_probs.items()}
        
        return dart_probs
    
    def analyze_checkout_statistics(self, probabilities):
        """Analyze and display checkout statistics"""
        print(f"\n📈 Checkout Analysis Results")
        print("=" * 60)
        
        # Find best and worst checkout scores
        scores_by_total_prob = sorted(probabilities.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print(f"\n🎯 Easiest Checkouts (Highest Probability):")
        for i, (score, data) in enumerate(scores_by_total_prob[:10]):
            total_ways = data['ways_1'] + data['ways_2'] + data['ways_3']
            print(f"   {i+1:2d}. Score {score:3d}: {data['total']:.4f} probability ({total_ways} total ways)")
        
        print(f"\n💀 Hardest Checkouts (Lowest Probability):")
        for i, (score, data) in enumerate(scores_by_total_prob[-10:]):
            total_ways = data['ways_1'] + data['ways_2'] + data['ways_3']
            print(f"   {i+1:2d}. Score {score:3d}: {data['total']:.6f} probability ({total_ways} total ways)")
        
        # Analyze by dart count
        print(f"\n🎲 Analysis by Number of Darts:")
        
        # 1-dart checkouts
        one_dart_checkouts = [(score, data) for score, data in probabilities.items() if data['ways_1'] > 0]
        print(f"   1-dart checkouts possible: {len(one_dart_checkouts)} scores")
        if one_dart_checkouts:
            best_1_dart = max(one_dart_checkouts, key=lambda x: x[1]['1_dart'])
            print(f"   Best 1-dart: Score {best_1_dart[0]} ({best_1_dart[1]['1_dart']:.4f} probability)")
        
        # 2-dart checkouts
        avg_2_dart_prob = np.mean([data['2_dart'] for data in probabilities.values() if data['2_dart'] > 0])
        print(f"   Average 2-dart checkout probability: {avg_2_dart_prob:.6f}")
        
        # 3-dart checkouts
        avg_3_dart_prob = np.mean([data['3_dart'] for data in probabilities.values() if data['3_dart'] > 0])
        print(f"   Average 3-dart checkout probability: {avg_3_dart_prob:.8f}")
        
        # Impossible checkouts
        impossible_scores = [score for score, data in probabilities.items() if data['total'] == 0]
        print(f"\n❌ Impossible checkout scores: {impossible_scores}")
        
        return scores_by_total_prob
    
    def plot_checkout_probabilities(self, probabilities, skill_model='uniform'):
        """Create visualizations of checkout probabilities"""
        scores = list(range(2, 171))
        total_probs = [probabilities[score]['total'] for score in scores]
        prob_1_dart = [probabilities[score]['1_dart'] for score in scores]
        prob_2_dart = [probabilities[score]['2_dart'] for score in scores]
        prob_3_dart = [probabilities[score]['3_dart'] for score in scores]
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Total checkout probability
        ax1.plot(scores, total_probs, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(scores, total_probs, alpha=0.3)
        ax1.set_xlabel('Remaining Score')
        ax1.set_ylabel('Checkout Probability')
        ax1.set_title(f'Total Checkout Probability\n(Skill Model: {skill_model})')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(2, 170)
        
        # Plot 2: Checkout probability by dart count (stacked)
        ax2.fill_between(scores, prob_1_dart, alpha=0.8, label='1-dart', color='red')
        ax2.fill_between(scores, np.array(prob_1_dart) + np.array(prob_2_dart), prob_1_dart, 
                        alpha=0.8, label='2-dart', color='orange')
        ax2.fill_between(scores, np.array(prob_1_dart) + np.array(prob_2_dart) + np.array(prob_3_dart), 
                        np.array(prob_1_dart) + np.array(prob_2_dart), 
                        alpha=0.8, label='3-dart', color='green')
        ax2.set_xlabel('Remaining Score')
        ax2.set_ylabel('Checkout Probability')
        ax2.set_title('Checkout Probability by Dart Count (Stacked)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(2, 170)
        
        # Plot 3: Number of ways to checkout
        ways_1_dart = [probabilities[score]['ways_1'] for score in scores]
        ways_2_dart = [probabilities[score]['ways_2'] for score in scores]
        ways_3_dart = [probabilities[score]['ways_3'] for score in scores]
        
        ax3.bar(scores, ways_1_dart, alpha=0.8, label='1-dart ways', color='red', width=0.8)
        ax3.bar(scores, ways_2_dart, bottom=ways_1_dart, alpha=0.8, label='2-dart ways', color='orange', width=0.8)
        ax3.bar(scores, ways_3_dart, bottom=np.array(ways_1_dart) + np.array(ways_2_dart), 
               alpha=0.8, label='3-dart ways', color='green', width=0.8)
        ax3.set_xlabel('Remaining Score')
        ax3.set_ylabel('Number of Ways to Checkout')
        ax3.set_title('Number of Checkout Combinations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 172)
        
        # Plot 4: Heatmap of checkout difficulty
        # Reshape data for heatmap (10 rows x 17 columns to cover scores 2-170)
        heatmap_data = []
        for row in range(10):
            row_data = []
            for col in range(17):
                score = row * 17 + col + 2
                if score <= 170:
                    row_data.append(probabilities[score]['total'])
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax4.set_title('Checkout Difficulty Heatmap\n(Green = Easy, Red = Hard)')
        ax4.set_xlabel('Score (within row)')
        ax4.set_ylabel('Score Range')
        
        # Add score labels to heatmap
        for row in range(10):
            for col in range(17):
                score = row * 17 + col + 2
                if score <= 170:
                    text = ax4.text(col, row, str(score), ha="center", va="center", 
                                  color="black" if heatmap_data[row][col] > 0.0001 else "white", 
                                  fontsize=6)
        
        plt.colorbar(im, ax=ax4, label='Checkout Probability')
        
        plt.tight_layout()
        plt.savefig(f'/Users/a_fin/Documents/Stat_of_darts/checkout_analysis_{skill_model}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def find_famous_checkouts(self, probabilities):
        """Analyze famous/notable checkout finishes"""
        print(f"\n🏆 Famous Checkout Analysis:")
        print("-" * 40)
        
        famous_scores = [170, 167, 164, 161, 158, 156, 155, 152, 150, 120, 100, 81, 9]
        
        for score in famous_scores:
            if score in probabilities:
                data = probabilities[score]
                ways = self.checkout_combinations[score]
                total_ways = data['ways_1'] + data['ways_2'] + data['ways_3']
                
                print(f"\nScore {score}:")
                print(f"   Total probability: {data['total']:.6f}")
                print(f"   Total ways: {total_ways}")
                
                if ways['1_dart']:
                    print(f"   1-dart: {ways['1_dart']}")
                if ways['2_dart'][:3]:  # Show first 3 for brevity
                    print(f"   2-dart (sample): {ways['2_dart'][:3]}")
                if ways['3_dart'][:3]:  # Show first 3 for brevity
                    print(f"   3-dart (sample): {ways['3_dart'][:3]}")
    
    def save_results(self, probabilities, skill_model='uniform'):
        """Save detailed results to JSON file"""
        filename = f'/Users/a_fin/Documents/Stat_of_darts/checkout_probabilities_{skill_model}.json'
        
        # Convert to serializable format
        results = {
            'skill_model': skill_model,
            'probabilities': probabilities,
            'checkout_combinations': self.checkout_combinations
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Saved detailed results to {filename}")

def main():
    """Main function to run checkout analysis"""
    analyzer = CheckoutAnalyzer()
    
    print("🎯 Dartboard Checkout Probability Analysis")
    print("=" * 50)
    print(f"Analyzing checkout probabilities for scores 2-170")
    print(f"Using double-out rules (must finish on double)")
    
    # Analyze with uniform skill model
    print(f"\n🎲 Uniform Skill Model (all dart scores equally likely):")
    uniform_probs = analyzer.calculate_checkout_probabilities('uniform')
    uniform_stats = analyzer.analyze_checkout_statistics(uniform_probs)
    analyzer.plot_checkout_probabilities(uniform_probs, 'uniform')
    analyzer.find_famous_checkouts(uniform_probs)
    analyzer.save_results(uniform_probs, 'uniform')
    
    # Analyze with skilled player model
    print(f"\n🏹 Skilled Player Model (higher scores more likely):")
    skilled_probs = analyzer.calculate_checkout_probabilities('skilled')
    skilled_stats = analyzer.analyze_checkout_statistics(skilled_probs)
    analyzer.plot_checkout_probabilities(skilled_probs, 'skilled')
    analyzer.find_famous_checkouts(skilled_probs)
    analyzer.save_results(skilled_probs, 'skilled')
    
    print(f"\n✅ Checkout analysis complete!")
    print(f"📁 Results saved to /Users/a_fin/Documents/Stat_of_darts/")

if __name__ == "__main__":
    main()