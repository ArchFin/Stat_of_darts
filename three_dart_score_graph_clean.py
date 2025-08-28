import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import csv

class ThreeDartScoreAnalyzer:
    def __init__(self):
        # All possible single dart scores on a dartboard
        self.possible_scores = self.generate_all_possible_scores()
        
    def generate_all_possible_scores(self):
        """Generate all possible single dart scores (excluding board miss)"""
        scores = set()
        
        # Single scores: 1-20
        for i in range(1, 21):
            scores.add(i)
        
        # Double scores: 2, 4, 6, ..., 40
        for i in range(1, 21):
            scores.add(i * 2)
            
        # Triple scores: 3, 6, 9, ..., 60
        for i in range(1, 21):
            scores.add(i * 3)
            
        # Bulls
        scores.add(25)  # Outer bull
        scores.add(50)  # Inner bull
        
        return sorted(list(scores))
    
    def calculate_three_dart_combinations(self):
        """Calculate all possible 3-dart combinations and their frequencies"""
        combinations = defaultdict(list)
        total_combinations = 0
        
        # Generate all possible 3-dart combinations (with replacement)
        for dart1 in self.possible_scores:
            for dart2 in self.possible_scores:
                for dart3 in self.possible_scores:
                    total_score = dart1 + dart2 + dart3
                    combinations[total_score].append((dart1, dart2, dart3))
                    total_combinations += 1
        
        # Calculate probabilities (assuming uniform distribution)
        probabilities = {}
        for total_score, combo_list in combinations.items():
            probabilities[total_score] = len(combo_list) / total_combinations
            
        return combinations, probabilities, total_combinations
    
    def create_big_network_graph(self, min_probability=0.001, max_nodes=50):
        """Create one large network graph showing dart score relationships"""
        combinations, probabilities, total_combinations = self.calculate_three_dart_combinations()
        
        print(f"📊 Found {len(probabilities)} unique total scores")
        print(f"🎯 Creating network with top {max_nodes} most probable scores")
        
        # Filter to most probable scores to keep graph readable
        filtered_scores = {score: prob for score, prob in probabilities.items() 
                          if prob >= min_probability}
        
        # Take top N scores by probability
        top_scores = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        focus_scores = [score for score, _ in top_scores]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes (total scores) with probability as attribute
        for score, prob in top_scores:
            G.add_node(score, probability=prob, type="total", combinations=len(combinations[score]))
        
        # Add edges showing how individual darts contribute to totals
        edge_weights = defaultdict(int)
        
        for total_score in focus_scores:
            if total_score in combinations:
                combo_list = combinations[total_score]
                
                # For each combination, create edges from individual dart scores to the total
                for dart1, dart2, dart3 in combo_list:
                    for dart_score in [dart1, dart2, dart3]:
                        edge_key = (dart_score, total_score)
                        edge_weights[edge_key] += 1
        
        # Add edges with weights (also ensures dart score nodes are added)
        for (dart_score, total_score), weight in edge_weights.items():
            if dart_score != total_score and total_score in focus_scores:
                norm_weight = weight / len(combinations[total_score]) if total_score in combinations else 0
                if norm_weight > 0.01:
                    # Add dart score node if missing
                    if dart_score not in G:
                        G.add_node(dart_score, type="dart")
                    G.add_edge(dart_score, total_score, weight=norm_weight, count=weight)
        
        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(20, 16))
        
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Separate total score vs dart score nodes
        total_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "total"]
        dart_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "dart"]
        
        # Total score node sizes/colors
        total_sizes = [probabilities[n] * 50000 for n in total_nodes]
        total_colors = [probabilities[n] for n in total_nodes]
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=total_nodes,
            node_size=total_sizes,
            node_color=total_colors,
            cmap="viridis",
            alpha=0.85,
            ax=ax
        )
        
        # Dart score nodes (uniform style)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=dart_nodes,
            node_size=300,
            node_color="lightgray",
            alpha=0.6,
            ax=ax
        )
        
        # Draw edges
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=[w*5 for w in edge_weights],
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            ax=ax
        )
        
        # Labels only for significant total scores
        significant_nodes = {n: n for n in total_nodes if probabilities[n] > 0.005 or n in [180, 120, 60]}
        nx.draw_networkx_labels(G, pos, significant_nodes, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Network Graph of 3-Dart Score Relationships\n'
                     'Node Size = Probability, Edge Width = Contribution Frequency', 
                     fontsize=16, fontweight='bold')
        
        # Colorbar for probabilities
        if len(total_colors) > 0:
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=min(total_colors), vmax=max(total_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Probability of Total Score', fontweight='bold')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=6, label='Dart Score'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                      markersize=8, label='Total Score (size ∝ probability)'),
            plt.Line2D([0], [0], color='gray', linewidth=2, 
                      label='Contribution (width ∝ frequency)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('/Users/a_fin/Documents/Stat_of_darts/big_network_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # --- Stats ---
        print(f"\n🕸️ Network Graph Statistics:")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        if G.number_of_nodes() > 0:
            print(f"   Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        
        if G.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(G)
            most_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n   Most connected nodes:")
            for node, centrality in most_connected:
                if node in probabilities:
                    prob = probabilities[node]
                    print(f"     {node}: {centrality:.3f} centrality, {prob:.4f} probability")
                else:
                    print(f"     {node}: {centrality:.3f} centrality (dart score)")
        
        print(f"\n🧠 Understanding the Results:")
        print(f"   • Total score nodes are larger/more colorful when more probable")
        print(f"   • Dart score nodes are smaller, gray, and feed into totals")
        print(f"   • Thicker edges = more frequent contribution of a dart score to a total")
        
        # Save CSV of totals
        csv_path = '/Users/a_fin/Documents/Stat_of_darts/three_dart_totals.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['TotalScore', 'Probability', 'NumCombinations'])
            for score in sorted(probabilities.keys()):
                writer.writerow([score, probabilities[score], len(combinations[score])])
        print(f"\n📁 Saved all 3-dart totals and probabilities to {csv_path}")
        
        return G
    
    def analyze_network_insights(self):
        """Provide detailed analysis and insights about the 3-dart scoring network"""
        combinations, probabilities, total_combinations = self.calculate_three_dart_combinations()
        
        print(f"\n🔍 Deep Analysis of 3-Dart Scoring:")
        print("=" * 60)
        
        # Find most and least probable scores
        sorted_by_prob = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        most_probable = sorted_by_prob[:5]
        least_probable = sorted_by_prob[-5:]
        
        print(f"\n📈 Most Probable 3-Dart Totals:")
        for i, (score, prob) in enumerate(most_probable):
            ways = len(combinations[score])
            print(f"   {i+1}. Score {score}: {prob:.4f} probability ({ways} ways)")
        
        print(f"\n📉 Least Probable 3-Dart Totals:")
        for i, (score, prob) in enumerate(least_probable):
            ways = len(combinations[score])
            print(f"   {i+1}. Score {score}: {prob:.6f} probability ({ways} ways)")
        
        # Analyze dart score versatility
        dart_versatility = defaultdict(set)
        for total_score, combo_list in combinations.items():
            for dart1, dart2, dart3 in combo_list:
                for dart_score in [dart1, dart2, dart3]:
                    dart_versatility[dart_score].add(total_score)
        
        # Most versatile dart scores
        versatility_ranking = sorted(dart_versatility.items(), 
                                   key=lambda x: len(x[1]), reverse=True)
        
        print(f"\n🎯 Most Versatile Single Dart Scores:")
        for i, (dart_score, totals) in enumerate(versatility_ranking[:10]):
            print(f"   {i+1}. Dart {dart_score}: contributes to {len(totals)} different 3-dart totals")
        
        # Perfect game analysis
        perfect_combinations = combinations.get(180, [])
        print(f"\n🏆 Perfect Game (180) Analysis:")
        print(f"   Total ways to score 180: {len(perfect_combinations)}")
        
        if perfect_combinations:
            unique_180_combos = Counter()
            for combo in perfect_combinations:
                sorted_combo = tuple(sorted(combo))
                unique_180_combos[sorted_combo] += 1
            
            print(f"   Unique combinations: {len(unique_180_combos)}")
            print(f"   Top ways to score 180:")
            for i, (combo, count) in enumerate(unique_180_combos.most_common(5)):
                d1, d2, d3 = combo
                percentage = (count / len(perfect_combinations)) * 100
                print(f"     {i+1}. {d1} + {d2} + {d3} → {count} ways ({percentage:.1f}%)")

def main():
    """Main function to run the three-dart score analysis"""
    analyzer = ThreeDartScoreAnalyzer()
    
    print("🎯 Three-Dart Score Network Analysis")
    print("=" * 50)
    print(f"Possible single dart scores: {len(analyzer.possible_scores)}")
    print(f"Score range: {min(analyzer.possible_scores)} to {max(analyzer.possible_scores)}")
    
    # Create big network graph
    print("\n🌐 Creating comprehensive network graph...")
    G = analyzer.create_big_network_graph(min_probability=0.0005, max_nodes=75)
    
    # Provide detailed analysis
    analyzer.analyze_network_insights()
    
    print(f"\n✅ Analysis complete! Network graph saved to /Users/a_fin/Documents/Stat_of_darts/")

if __name__ == "__main__":
    main()