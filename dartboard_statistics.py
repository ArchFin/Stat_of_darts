import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import Counter
import time

class DartboardStatistics:
    def __init__(self):
        # Official dartboard layout (clockwise from top)
        self.sectors = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        
        # Dartboard dimensions (in mm)
        self.outer_bull_radius = 16 / 2  # Outer bull radius
        self.inner_bull_radius = 6.35 / 2  # Inner bull radius
        self.triple_inner_radius = 99 / 2  # Inner edge of triple ring
        self.triple_outer_radius = 107 / 2 # Outer edge of triple ring
        self.double_inner_radius = 162 / 2 # Inner edge of double ring
        self.double_outer_radius = 170 / 2 # Outer edge of double ring
        
    def get_sector_angle(self, angle_degrees):
        """Convert angle to sector number (0-19)"""
        angle_degrees = angle_degrees % 360
        sector_angle = (angle_degrees + 9) % 360  # Offset by 9 degrees so 20 is at top
        sector_index = int(sector_angle / 18)
        return sector_index
    
    def calculate_score(self, distance, angle_degrees):
        """Calculate dart score based on distance from center and angle"""
        sector_index = self.get_sector_angle(angle_degrees)
        sector_value = self.sectors[sector_index]
        
        if distance <= self.inner_bull_radius:
            return 50  # Inner bull
        elif distance <= self.outer_bull_radius:
            return 25  # Outer bull
        elif self.triple_inner_radius <= distance <= self.triple_outer_radius:
            return sector_value * 3  # Triple
        elif self.double_inner_radius <= distance <= self.double_outer_radius:
            return sector_value * 2  # Double
        elif distance <= self.double_outer_radius:
            return sector_value  # Single
        else:
            return 0  # Miss (outside dartboard)
    
    def throw_dart(self):
        """Simulate throwing a dart - returns score"""
        # Uniform random distribution over the dartboard area
        max_radius = self.double_outer_radius
        distance = np.sqrt(np.random.uniform(0, 1)) * max_radius
        angle = random.uniform(0, 360)
        
        return self.calculate_score(distance, angle)
    
    def simulate_three_dart_round(self):
        """Simulate one round of three dart throws and return total score"""
        return sum(self.throw_dart() for _ in range(3))
    
    def run_simulation(self, num_rounds=1_000_000):
        """Run the complete simulation of multiple rounds"""
        print(f"🎯 Dartboard Statistics Simulation")
        print(f"Simulating {num_rounds:,} rounds of three dart throws...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Store all round totals
        round_totals = []
        
        # Progress tracking
        checkpoint = num_rounds // 10  # Update every 10%
        
        for round_num in range(num_rounds):
            total_score = self.simulate_three_dart_round()
            round_totals.append(total_score)
            
            # Progress update
            if (round_num + 1) % checkpoint == 0:
                progress = ((round_num + 1) / num_rounds) * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:5.1f}% ({round_num + 1:,} rounds) - {elapsed:.1f}s elapsed")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ Simulation completed in {total_time:.2f} seconds")
        print(f"Average speed: {num_rounds/total_time:,.0f} rounds per second")
        
        return round_totals

    def save_totals(self, round_totals, filename="three_dart_totals.npy"):
        """Save the list of round totals to a .npy file"""
        np.save(filename, np.array(round_totals))
        print(f"\n💾 Saved round totals to {filename}")
    
    def analyze_results(self, round_totals):
        """Analyze and display statistics from the simulation results"""
        print(f"\n📊 Statistical Analysis")
        print("=" * 40)
        
        # Basic statistics
        mean_score = np.mean(round_totals)
        median_score = np.median(round_totals)
        std_score = np.std(round_totals)
        min_score = min(round_totals)
        max_score = max(round_totals)
        
        print(f"Mean score:      {mean_score:.2f}")
        print(f"Median score:    {median_score:.2f}")
        print(f"Std deviation:   {std_score:.2f}")
        print(f"Minimum score:   {min_score}")
        print(f"Maximum score:   {max_score}")
        
        # Percentiles
        percentiles = [10, 25, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(round_totals, p)
            print(f"  {p:2d}th percentile: {value:.1f}")
        
        # Score frequency analysis
        score_counts = Counter(round_totals)
        most_common_scores = score_counts.most_common(10)
        
        print(f"\nMost common total scores:")
        for score, count in most_common_scores:
            percentage = (count / len(round_totals)) * 100
            print(f"  Score {score:3d}: {count:6,} times ({percentage:.2f}%)")
        
        # Special achievements
        perfect_rounds = sum(1 for score in round_totals if score >= 180)  # Perfect or near-perfect
        zero_rounds = sum(1 for score in round_totals if score == 0)
        high_scores = sum(1 for score in round_totals if score >= 100)
        
        print(f"\nSpecial achievements:")
        print(f"  Scores ≥ 180:   {perfect_rounds:6,} ({perfect_rounds/len(round_totals)*100:.3f}%)")
        print(f"  Scores ≥ 100:   {high_scores:6,} ({high_scores/len(round_totals)*100:.2f}%)")
        print(f"  Zero scores:     {zero_rounds:6,} ({zero_rounds/len(round_totals)*100:.3f}%)")
        
        return {
            'mean': mean_score,
            'median': median_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'most_common': most_common_scores
        }
    
    def plot_histogram(self, round_totals, stats):
        """Create a histogram of the score distribution and save to files"""
        plt.figure(figsize=(14, 8))
        
        # Create histogram
        bins = range(0, max(round_totals) + 5, 5)  # Bins of 5 points
        n, bins, patches = plt.hist(round_totals, bins=bins, alpha=0.7, color='skyblue', 
                                   edgecolor='black', linewidth=0.5)
        
        # Add statistics lines
        plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {stats["mean"]:.1f}')
        plt.axvline(stats['median'], color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {stats["median"]:.1f}')
        
        # Formatting
        plt.xlabel('Total Score (3 darts)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(f'Distribution of Three-Dart Total Scores\n({len(round_totals):,} simulated rounds)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics box
        stats_text = f"""Statistics:
Mean: {stats['mean']:.2f}
Std Dev: {stats['std']:.2f}
Min: {stats['min']}
Max: {stats['max']}"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig("/Users/a_fin/Documents/Stat_of_darts/three_dart_histogram.png")
        plt.show()

        # Probability and CDF plot
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.hist(round_totals, bins=bins, density=True, alpha=0.7, color='lightcoral', 
                edgecolor='black', linewidth=0.5)
        plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
        plt.xlabel('Total Score')
        plt.ylabel('Probability Density')
        plt.title('Probability Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.subplot(1, 2, 2)
        sorted_scores = np.sort(round_totals)
        cumulative_prob = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        plt.plot(sorted_scores, cumulative_prob, color='purple', linewidth=2)
        plt.xlabel('Total Score')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("/Users/a_fin/Documents/Stat_of_darts/three_dart_probability_cdf.png")
        plt.show()

def main():
    """Main function to run the dartboard statistics simulation"""
    dartboard_stats = DartboardStatistics()
    
    # Run the simulation
    num_rounds = 1_000_000
    round_totals = dartboard_stats.run_simulation(num_rounds)
    
    # Save results to file
    dartboard_stats.save_totals(round_totals, "three_dart_totals.npy")
    
    # Analyze results
    stats = dartboard_stats.analyze_results(round_totals)
    
    # Create visualizations
    print(f"\n📈 Generating histograms...")
    dartboard_stats.plot_histogram(round_totals, stats)

if __name__ == "__main__":
    main()