import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math

class DartboardSimulator:
    def __init__(self):
        # Official dartboard layout (clockwise from top)
        self.sectors = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        
        self.outer_bull_radius = 16 / 2  # Outer bull radius
        self.inner_bull_radius = 6.35 / 2  # Inner bull radius
        self.triple_inner_radius = 99 / 2  # Inner edge of triple ring
        self.triple_outer_radius = 107 / 2 # Outer edge of triple ring
        self.double_inner_radius = 162 / 2 # Inner edge of double ring
        self.double_outer_radius = 170 / 2 # Outer edge of double ring
        
        # Scale factor for visualization
        self.scale = 2
        
    def get_sector_angle(self, angle_degrees):
        """Convert angle to sector number (0-19)"""
        # Normalize angle to 0-360
        angle_degrees = angle_degrees % 360
        # Dartboard sectors are 18 degrees each, starting from 6 degrees offset
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
        """Simulate throwing a dart - returns (distance, angle, score)"""
        # Uniform random distribution over the dartboard area
        max_radius = self.double_outer_radius
        distance = np.sqrt(np.random.uniform(0, 1)) * max_radius
        angle = random.uniform(0, 360)
        
        score = self.calculate_score(distance, angle)
        return distance, angle, score
    
    def throw_three_darts(self):
        """Simulate throwing three darts"""
        throws = []
        total_score = 0
        
        print("🎯 Dartboard Simulation - Three Dart Throws")
        print("=" * 45)
        
        for i in range(3):
            distance, angle, score = self.throw_dart()
            throws.append((distance, angle, score))
            total_score += score
            
            # Convert to Cartesian coordinates for display
            x = distance * math.cos(math.radians(angle))
            y = distance * math.sin(math.radians(angle))
            
            print(f"Dart {i+1}:")
            print(f"  Position: ({x:.1f}, {y:.1f}) mm from center")
            print(f"  Distance: {distance:.1f} mm")
            print(f"  Angle: {angle:.1f}°")
            print(f"  Score: {score} points")
            print()
        
        print(f"Total Score: {total_score} points")
        return throws, total_score
    
    def plot_dartboard(self, throws=None):
        """Plot the dartboard with optional dart throws"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw dartboard rings
        circles = [
            (self.double_outer_radius * self.scale, 'black', 2),
            (self.double_inner_radius * self.scale, 'red' if True else 'green', 1),
            (self.triple_outer_radius * self.scale, 'red' if True else 'green', 1),
            (self.triple_inner_radius * self.scale, 'black', 1),
            (self.outer_bull_radius * self.scale, 'red', 1),
            (self.inner_bull_radius * self.scale, 'black', 1),
        ]
        
        for radius, color, linewidth in circles:
            circle = plt.Circle((0, 0), radius, fill=False, color=color, linewidth=linewidth)
            ax.add_patch(circle)
        
        # Fill bull areas
        outer_bull = plt.Circle((0, 0), self.outer_bull_radius * self.scale, color='green', alpha=0.7)
        inner_bull = plt.Circle((0, 0), self.inner_bull_radius * self.scale, color='red', alpha=0.7)
        ax.add_patch(outer_bull)
        ax.add_patch(inner_bull)
        
        # Draw sector lines and numbers
        for i, sector_value in enumerate(self.sectors):
            angle = i * 18 - 9  # Convert sector index to angle
            angle_rad = math.radians(angle)
            
            # Draw sector line
            x_end = (self.double_outer_radius * self.scale) * math.cos(angle_rad)
            y_end = (self.double_outer_radius * self.scale) * math.sin(angle_rad)
            ax.plot([0, x_end], [0, y_end], 'k-', linewidth=0.5, alpha=0.7)
            
            # Add sector numbers
            text_radius = (self.double_outer_radius + 10) * self.scale
            text_x = text_radius * math.cos(angle_rad + math.radians(9))  # Center of sector
            text_y = text_radius * math.sin(angle_rad + math.radians(9))
            ax.text(text_x, text_y, str(sector_value), ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Draw triple and double rings with alternating colors
        for i in range(20):
            start_angle = i * 18 - 9
            end_angle = (i + 1) * 18 - 9
            color = 'red' if i % 2 == 0 else 'green'
            
            # Triple ring
            triple_wedge = patches.Wedge((0, 0), self.triple_outer_radius * self.scale,
                                       start_angle, end_angle, 
                                       width=(self.triple_outer_radius - self.triple_inner_radius) * self.scale,
                                       facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.add_patch(triple_wedge)
            
            # Double ring
            double_wedge = patches.Wedge((0, 0), self.double_outer_radius * self.scale,
                                       start_angle, end_angle,
                                       width=(self.double_outer_radius - self.double_inner_radius) * self.scale,
                                       facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.add_patch(double_wedge)
        
        # Plot dart throws
        if throws:
            colors = ['blue', 'orange', 'purple']
            for i, (distance, angle, score) in enumerate(throws):
                x = distance * math.cos(math.radians(angle)) * self.scale
                y = distance * math.sin(math.radians(angle)) * self.scale
                
                ax.plot(x, y, 'o', color=colors[i], markersize=8, markeredgecolor='black', 
                       markeredgewidth=2, label=f'Dart {i+1}: {score} pts')
                
                # Add score annotation
                ax.annotate(f'{score}', (x, y), xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                           fontsize=10, fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Dartboard Simulation', fontsize=16, fontweight='bold')
        
        if throws:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the dartboard simulation"""
    dartboard = DartboardSimulator()
    
    # Simulate three dart throws
    throws, total_score = dartboard.throw_three_darts()
    
    # Plot the dartboard with throws
    print("\nGenerating dartboard visualization...")
    dartboard.plot_dartboard(throws)

if __name__ == "__main__":
    main()