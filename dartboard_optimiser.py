import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import math
from scipy.stats import multivariate_normal
import time
from multiprocessing import Pool, cpu_count
from functools import partial

class DartboardOptimiser:
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
        
        # Scale factor for visualisation
        self.scale = 2
        
    def get_sector_angle(self, angle_degrees):
        """Convert angle to sector number (0-19)"""
        angle_degrees = angle_degrees % 360
        sector_angle = (angle_degrees + 9) % 360  # Offset by 9 degrees so 20 is at top
        sector_index = int(sector_angle / 18)
        sector_index = max(0, min(sector_index, len(self.sectors) - 1))  # Ensure index is within valid range
        return sector_index
    
    def calculate_score(self, x, y):
        """Calculate dart score based on Cartesian coordinates"""
        distance = math.sqrt(x**2 + y**2)
        
        # Check if dart is outside the board
        if distance > self.double_outer_radius:
            return 0
        
        # Calculate angle in degrees (0 = right, 90 = up)
        angle_rad = math.atan2(y, x)
        angle_degrees = math.degrees(angle_rad)
        if angle_degrees < 0:
            angle_degrees += 360
        
        # Adjust angle to match dartboard orientation (90 degrees = top)
        # Convert from math convention (0° = right) to dartboard convention (0° = top)
        angle_degrees = (90 - angle_degrees) % 360
        
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
        else:
            return sector_value  # Single
    
    def simulate_throws_at_point(self, aim_x, aim_y, accuracy_std, num_throws=100000):
        """Simulate many dart throws aimed at a specific point"""
        # Generate random throws using 2D normal distribution
        throws_x = np.random.normal(aim_x, accuracy_std, num_throws)
        throws_y = np.random.normal(aim_y, accuracy_std, num_throws)
        
        # Calculate scores for all throws
        scores = []
        for x, y in zip(throws_x, throws_y):
            score = self.calculate_score(x, y)
            scores.append(score)
        
        return np.mean(scores)
    
    def process_aim_point(self, point_data):
        """Process a single aim point for parallel computation"""
        aim_x, aim_y, accuracy_std, num_throws, max_distance = point_data
        
        # Only test points within reasonable range of the board
        distance_from_center = math.sqrt(aim_x**2 + aim_y**2)
        if distance_from_center <= max_distance:
            return self.simulate_throws_at_point(aim_x, aim_y, accuracy_std, num_throws)
        else:
            return 0.0  # No point aiming far outside the board
    
    def create_aim_grid(self, grid_size=25):
        """Create a grid of aim points covering the dartboard"""
        # Create grid from -100 to 100 mm (covering the dartboard)
        max_coord = self.double_outer_radius * 1.1
        x_coords = np.linspace(-max_coord, max_coord, grid_size)
        y_coords = np.linspace(-max_coord, max_coord, grid_size)
        
        return np.meshgrid(x_coords, y_coords)
    
    def optimise_dartboard_coarse_fine(self, accuracy_std, coarse_grid_size=50, fine_grid_size=21, 
                                       coarse_throws=10000, fine_throws=50000, num_candidates=5, 
                                       fine_radius=15.0):
        """Two-stage coarse-to-fine optimization for conclusive results"""
        print(f"🎯 Two-Stage Dartboard Optimisation Analysis")
        print(f"Accuracy (std dev): {accuracy_std:.1f} mm")
        print(f"Stage 1 - Coarse: {coarse_grid_size}x{coarse_grid_size} = {coarse_grid_size**2} points, {coarse_throws:,} throws each")
        print(f"Stage 2 - Fine: {num_candidates} regions, {fine_grid_size}x{fine_grid_size} = {fine_grid_size**2} points each, {fine_throws:,} throws each")
        print(f"Using {cpu_count()} CPU cores for parallel processing")
        print("=" * 80)
        
        overall_start = time.time()
        
        # Stage 1: Coarse search
        print(f"🔍 Stage 1: Coarse Search")
        print("-" * 40)
        
        coarse_start = time.time()
        X_coarse, Y_coarse = self.create_aim_grid(coarse_grid_size)
        expected_scores_coarse = np.zeros_like(X_coarse)
        
        # Only evaluate points within the dartboard region
        max_distance = self.double_outer_radius * 1.1
        coarse_points = []
        valid_indices = []
        
        for i in range(coarse_grid_size):
            for j in range(coarse_grid_size):
                aim_x = X_coarse[i, j]
                aim_y = Y_coarse[i, j]
                distance = math.sqrt(aim_x**2 + aim_y**2)
                if distance <= max_distance:
                    coarse_points.append((aim_x, aim_y, accuracy_std, coarse_throws, max_distance))
                    valid_indices.append((i, j))
        
        print(f"Evaluating {len(coarse_points)} valid points (filtered from {coarse_grid_size**2} total)")
        
        with Pool(processes=cpu_count()) as pool:
            process_func = partial(self._process_aim_point_wrapper, self)
            coarse_results = pool.map(process_func, coarse_points)
        
        # Fill in results
        for idx, (i, j) in enumerate(valid_indices):
            expected_scores_coarse[i, j] = coarse_results[idx]
        
        coarse_time = time.time() - coarse_start
        print(f"✅ Coarse search completed in {coarse_time:.2f} seconds")
        
        # Find top candidates from coarse search
        flat_scores = expected_scores_coarse.flatten()
        flat_x = X_coarse.flatten()
        flat_y = Y_coarse.flatten()
        
        # Filter out zero scores (outside dartboard)
        valid_mask = flat_scores > 0
        if not np.any(valid_mask):
            print("❌ No valid points found in coarse search!")
            return X_coarse, Y_coarse, expected_scores_coarse, (0, 0, 0)
        
        valid_scores = flat_scores[valid_mask]
        valid_x = flat_x[valid_mask]
        valid_y = flat_y[valid_mask]
        
        top_indices = np.argsort(valid_scores)[-num_candidates:][::-1]
        candidates = [(valid_x[i], valid_y[i], valid_scores[i]) for i in top_indices]
        
        print(f"\nTop {len(candidates)} candidates from coarse search:")
        for i, (x, y, score) in enumerate(candidates):
            print(f"  {i+1}. ({x:6.1f}, {y:6.1f}) mm → {score:.2f} points")
        
        # Stage 2: Fine search around each candidate
        print(f"\n🔬 Stage 2: Fine Search")
        print("-" * 40)
        
        best_overall_score = 0
        best_overall_point = (0, 0, 0)
        all_fine_results = []
        
        for candidate_idx, (cand_x, cand_y, cand_score) in enumerate(candidates):
            print(f"\nRefining candidate {candidate_idx + 1}/{len(candidates)}: ({cand_x:.1f}, {cand_y:.1f})")
            
            fine_start = time.time()
            
            # Create fine grid around candidate
            x_min, x_max = cand_x - fine_radius, cand_x + fine_radius
            y_min, y_max = cand_y - fine_radius, cand_y + fine_radius
            
            x_fine = np.linspace(x_min, x_max, fine_grid_size)
            y_fine = np.linspace(y_min, y_max, fine_grid_size)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
            
            # Evaluate fine grid
            fine_points = []
            fine_valid_indices = []
            
            for i in range(fine_grid_size):
                for j in range(fine_grid_size):
                    aim_x = X_fine[i, j]
                    aim_y = Y_fine[i, j]
                    distance = math.sqrt(aim_x**2 + aim_y**2)
                    if distance <= max_distance:
                        fine_points.append((aim_x, aim_y, accuracy_std, fine_throws, max_distance))
                        fine_valid_indices.append((i, j))
            
            print(f"  Evaluating {len(fine_points)} fine points with {fine_throws:,} throws each...")
            
            with Pool(processes=cpu_count()) as pool:
                process_func = partial(self._process_aim_point_wrapper, self)
                fine_results = pool.map(process_func, fine_points)
            
            # Find best in this fine region
            if fine_results:
                best_fine_idx = np.argmax(fine_results)
                best_fine_i, best_fine_j = fine_valid_indices[best_fine_idx]
                best_fine_x = X_fine[best_fine_i, best_fine_j]
                best_fine_y = Y_fine[best_fine_i, best_fine_j]
                best_fine_score = fine_results[best_fine_idx]
                
                all_fine_results.append((best_fine_x, best_fine_y, best_fine_score, X_fine, Y_fine, fine_results, fine_valid_indices))
                
                if best_fine_score > best_overall_score:
                    best_overall_score = best_fine_score
                    best_overall_point = (best_fine_x, best_fine_y, best_fine_score)
                
                fine_time = time.time() - fine_start
                print(f"  ✅ Best in region: ({best_fine_x:.2f}, {best_fine_y:.2f}) → {best_fine_score:.3f} points ({fine_time:.1f}s)")
        
        overall_time = time.time() - overall_start
        
        print(f"\n🏆 Final Results:")
        print("=" * 50)
        print(f"Overall optimization completed in {overall_time:.2f} seconds")
        print(f"Best aim point: ({best_overall_point[0]:.2f}, {best_overall_point[1]:.2f}) mm")
        print(f"Expected score: {best_overall_point[2]:.3f} points per dart")
        
        # Return the coarse grid for visualization (fine results are more accurate but harder to visualize)
        return X_coarse, Y_coarse, expected_scores_coarse, best_overall_point

    def optimise_dartboard(self, accuracy_std, grid_size=25, num_throws=100000):
        """Legacy single-stage optimization (kept for backward compatibility)"""
        return self.optimise_dartboard_coarse_fine(
            accuracy_std, 
            coarse_grid_size=grid_size, 
            fine_grid_size=21,
            coarse_throws=num_throws//5,
            fine_throws=num_throws,
            num_candidates=3,
            fine_radius=10.0
        )
    
    @staticmethod
    def _process_aim_point_wrapper(optimiser_instance, point_data):
        """Static wrapper for parallel processing"""
        return optimiser_instance.process_aim_point(point_data)
    
    def plot_optimisation_results(self, X, Y, expected_scores, optimal_point, accuracy_std):
        """Create visualisations of the optimisation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Heatmap of expected scores
        im1 = ax1.contourf(X, Y, expected_scores, levels=50, cmap='viridis')
        ax1.contour(X, Y, expected_scores, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        
        # Draw dartboard outline and scoring zones
        zones = [
            (self.inner_bull_radius, 'red', 'Inner Bull'),
            (self.outer_bull_radius, 'green', 'Outer Bull'),
            (self.triple_inner_radius, 'blue', 'Triple Ring (Inner)'),
            (self.triple_outer_radius, 'blue', 'Triple Ring (Outer)'),
            (self.double_inner_radius, 'purple', 'Double Ring (Inner)'),
            (self.double_outer_radius, 'purple', 'Double Ring (Outer)'),
        ]

        for radius, color, label in zones:
            circle = plt.Circle((0, 0), radius, fill=False, color=color, linewidth=1, linestyle='--', label=label)
            ax1.add_patch(circle)
        ax1.legend()

        # Mark optimal point
        optimal_x, optimal_y, optimal_score = optimal_point
        ax1.plot(optimal_x, optimal_y, 'r*', markersize=15, markeredgecolor='white', 
                markeredgewidth=2, label=f'Optimal: ({optimal_x:.1f}, {optimal_y:.1f})')
        
        ax1.set_xlabel('X Position (mm)', fontweight='bold')
        ax1.set_ylabel('Y Position (mm)', fontweight='bold')
        ax1.set_title(f'Expected Score Heatmap\nAccuracy σ = {accuracy_std:.1f} mm', 
                     fontweight='bold')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Expected Score per Dart', fontweight='bold')
        
        # Plot 2: Dartboard with optimal aim point
        self.plot_dartboard_with_aim(ax2, optimal_x, optimal_y, accuracy_std)
        
        plt.tight_layout()
        plt.savefig(f"/Users/a_fin/Documents/Stat_of_darts/results/dartboard_optimisation_std{accuracy_std:.1f}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a separate 3D surface plot
        fig = plt.figure(figsize=(12, 9))
        ax3 = fig.add_subplot(111, projection='3d')
        
        surf = ax3.plot_surface(X, Y, expected_scores, cmap='viridis', alpha=0.8)
        ax3.scatter([optimal_x], [optimal_y], [optimal_score], color='red', s=100, 
                   label=f'Optimal: {optimal_score:.2f}')
        
        ax3.set_xlabel('X Position (mm)')
        ax3.set_ylabel('Y Position (mm)')
        ax3.set_zlabel('Expected Score')
        ax3.set_title(f'3D Expected Score Surface\nAccuracy σ = {accuracy_std:.1f} mm')
        
        plt.colorbar(surf, ax=ax3, shrink=0.5)
        ax3.legend()
        
        plt.savefig(f"/Users/a_fin/Documents/Stat_of_darts/results/dartboard_3d_surface_std{accuracy_std:.1f}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_dartboard_with_aim(self, ax, aim_x, aim_y, accuracy_std):
        """Plot dartboard with aim point and accuracy visualisation"""
        # Draw dartboard rings
        circles = [
            (self.double_outer_radius, 'black', 2),
            (self.double_inner_radius, 'gray', 1),
            (self.triple_outer_radius, 'gray', 1),
            (self.triple_inner_radius, 'gray', 1),
            (self.outer_bull_radius, 'red', 1),
            (self.inner_bull_radius, 'black', 1),
        ]
        
        for radius, color, linewidth in circles:
            circle = plt.Circle((0, 0), radius, fill=False, color=color, linewidth=linewidth)
            ax.add_patch(circle)
        
        # Fill bull areas
        outer_bull = plt.Circle((0, 0), self.outer_bull_radius, color='green', alpha=0.3)
        inner_bull = plt.Circle((0, 0), self.inner_bull_radius, color='red', alpha=0.3)
        ax.add_patch(outer_bull)
        ax.add_patch(inner_bull)
        
        # Draw sector boundary lines
        for k in range(20):
            boundary_angle = 90 - 9 - k * 18  # Boundaries start at 81° and go clockwise
            boundary_rad = math.radians(boundary_angle)
            x_end = self.double_outer_radius * math.cos(boundary_rad)
            y_end = self.double_outer_radius * math.sin(boundary_rad)
            ax.plot([0, x_end], [0, y_end], 'k-', linewidth=0.5, alpha=0.5)

        # Add sector numbers derived from the scoring function to ensure alignment
        text_radius = self.double_outer_radius + 10
        # Place sector numbers from the canonical sectors list at sector centers
        for i, sector_value in enumerate(self.sectors):
            center_angle = 90 - i * 18  # Center of each sector (clockwise from top)
            center_rad = math.radians(center_angle)
            text_x = text_radius * math.cos(center_rad)
            text_y = text_radius * math.sin(center_rad)
            ax.text(text_x, text_y, str(sector_value), ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Mark aim point
        ax.plot(aim_x, aim_y, 'r*', markersize=12, markeredgecolor='black', 
               markeredgewidth=1, label=f'Aim Point')
        
        # Show accuracy circle (1 standard deviation)
        accuracy_circle = plt.Circle((aim_x, aim_y), accuracy_std, 
                                   fill=False, color='red', linewidth=2, linestyle='--',
                                   label=f'1σ accuracy ({accuracy_std:.1f} mm)')
        ax.add_patch(accuracy_circle)
        
        # Show 2 standard deviation circle
        accuracy_circle_2sig = plt.Circle((aim_x, aim_y), 2*accuracy_std, 
                                        fill=False, color='orange', linewidth=1, linestyle=':',
                                        label=f'2σ accuracy ({2*accuracy_std:.1f} mm)')
        ax.add_patch(accuracy_circle_2sig)
        
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (mm)', fontweight='bold')
        ax.set_ylabel('Y Position (mm)', fontweight='bold')
        ax.set_title(f'Optimal Aim Point\n({aim_x:.1f}, {aim_y:.1f}) mm', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add the verification check for triple 20
        self.check_visualisation(ax)
    
    def check_visualisation(self, ax):
        """Add a point to verify where the algorithm thinks triple 20 is"""
        # Triple 20 should be at the top (90 degrees in standard coordinates)
        triple_20_radius = (self.triple_inner_radius + self.triple_outer_radius) / 2
        
        # Calculate coordinates for triple 20 (at the top)
        x = 0  # At the top, x = 0
        y = triple_20_radius  # At the top, y = positive radius
        
        # Verify what the algorithm thinks this point scores
        score = self.calculate_score(x, y)
        
        ax.legend()
    
    def compare_accuracies(self, accuracy_levels, coarse_grid_size=60, fine_throws=100000):
        """Compare optimal strategies for different accuracy levels using coarse-to-fine optimization"""
        print(f"🎯 Comparing Different Accuracy Levels with Coarse-to-Fine Optimization")
        print("=" * 50)
        
        results = {}
        
        for accuracy in accuracy_levels:
            print(f"\nAnalysing accuracy level: {accuracy:.1f} mm")
            X, Y, expected_scores, optimal_point = self.optimise_dartboard_coarse_fine(
                accuracy, 
                coarse_grid_size=coarse_grid_size,
                fine_grid_size=21,
                coarse_throws=100000,
                fine_throws=fine_throws,
                num_candidates=5,
                fine_radius=10.0
            )
            results[accuracy] = optimal_point
            
            # Generate individual visualizations for each accuracy level
            self.plot_optimisation_results(X, Y, expected_scores, optimal_point, accuracy)
        
        # Plot comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw basic dartboard
        dartboard_circle = plt.Circle((0, 0), self.double_outer_radius, 
                                    fill=False, color='black', linewidth=2)
        ax.add_patch(dartboard_circle)
        
        # Plot optimal points for each accuracy
        colors = plt.cm.viridis(np.linspace(0, 1, len(accuracy_levels)))
        
        for i, (accuracy, (opt_x, opt_y, opt_score)) in enumerate(results.items()):
            ax.plot(opt_x, opt_y, 'o', color=colors[i], markersize=10, 
                   label=f'σ={accuracy:.1f}mm: ({opt_x:.1f}, {opt_y:.1f}) → {opt_score:.2f}pts')
        
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (mm)', fontweight='bold')
        ax.set_ylabel('Y Position (mm)', fontweight='bold')
        ax.set_title('Optimal Aim Points vs Accuracy Level', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig("/Users/a_fin/Documents/Stat_of_darts/results/accuracy_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return results

def main():
    """Main function to run dartboard optimisation"""
    optimiser = DartboardOptimiser()
    
    # Single accuracy analysis
#    accuracy_std = 15.0  # mm standard deviation (configurable)
#    X, Y, expected_scores, optimal_point = optimiser.optimise_dartboard(
#        accuracy_std, grid_size=40, num_throws=1000000)
    
    # Plot results
#    optimiser.plot_optimisation_results(X, Y, expected_scores, optimal_point, accuracy_std)
    
    # Compare different accuracy levels
    accuracy_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    comparison_results = optimiser.compare_accuracies(accuracy_levels)
    
    print(f"\n📊 Summary of Optimal Strategies:")
    print("=" * 50)
    for accuracy, (opt_x, opt_y, opt_score) in comparison_results.items():
        print(f"Accuracy σ={accuracy:4.1f}mm: Aim at ({opt_x:6.1f}, {opt_y:6.1f}) → {opt_score:.2f} pts/dart")

if __name__ == "__main__":
    main()