import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458    # Speed of light (m/s)
M_sun = 1.989e30 # Solar mass (kg)

class MassiveHand:
    """
    A class to simulate the gravitational effects of a massive hand in space
    """
    
    def __init__(self, total_mass=1e24, hand_size=0.2):
        """
        Initialize the massive hand
        
        Parameters:
        total_mass: Total mass of the hand in kg
        hand_size: Characteristic size of the hand in meters
        """
        self.total_mass = total_mass
        self.hand_size = hand_size
        self.mass_points = self._create_hand_mass_distribution()
        
    def _create_hand_mass_distribution(self):
        """
        Create a simplified mass distribution for a hand shape
        Returns array of (x, y, z, mass) points
        """
        points = []
        
        # Palm (rectangular region)
        palm_mass = 0.4 * self.total_mass
        palm_points = 20
        for i in range(palm_points):
            x = np.random.uniform(-0.3, 0.3) * self.hand_size
            y = np.random.uniform(-0.5, 0.1) * self.hand_size
            z = np.random.uniform(-0.1, 0.1) * self.hand_size
            points.append([x, y, z, palm_mass / palm_points])
        
        # Five fingers (cylindrical regions)
        finger_mass = 0.6 * self.total_mass / 5
        finger_positions = [
            (-0.25, 0.1, 0),  # Thumb
            (-0.1, 0.1, 0),   # Index
            (0, 0.1, 0),      # Middle
            (0.1, 0.1, 0),    # Ring
            (0.2, 0.1, 0)     # Pinky
        ]
        
        for fx, fy, fz in finger_positions:
            finger_points = 15
            for i in range(finger_points):
                x = fx * self.hand_size + np.random.uniform(-0.05, 0.05) * self.hand_size
                y = (fy + 0.3 * i / finger_points) * self.hand_size
                z = fz * self.hand_size + np.random.uniform(-0.02, 0.02) * self.hand_size
                points.append([x, y, z, finger_mass / finger_points])
        
        return np.array(points)
    
    def gravitational_potential(self, x, y, z):
        """
        Calculate gravitational potential at point (x, y, z)
        """
        potential = 0
        for point in self.mass_points:
            px, py, pz, mass = point
            r = np.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2)
            if r > 1e-10:  # Avoid division by zero
                potential -= G * mass / r
        return potential
    
    def gravitational_field(self, x, y, z):
        """
        Calculate gravitational field (acceleration) at point (x, y, z)
        Returns (ax, ay, az)
        """
        ax = ay = az = 0
        for point in self.mass_points:
            px, py, pz, mass = point
            dx, dy, dz = x - px, y - py, z - pz
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if r > 1e-10:
                r3 = r**3
                field_strength = G * mass / r3
                ax -= field_strength * dx
                ay -= field_strength * dy
                az -= field_strength * dz
        return ax, ay, az
    
    def schwarzschild_radius(self):
        """Calculate the Schwarzschild radius of the hand"""
        return 2 * G * self.total_mass / c**2
    
    def escape_velocity(self, distance):
        """Calculate escape velocity at given distance"""
        return np.sqrt(2 * G * self.total_mass / distance)
    
    def orbital_velocity(self, distance):
        """Calculate circular orbital velocity at given distance"""
        return np.sqrt(G * self.total_mass / distance)

def simulate_particle_orbit(hand, initial_pos, initial_vel, dt=0.1, steps=10000):
    """
    Simulate the orbit of a test particle around the massive hand
    """
    positions = []
    pos = np.array(initial_pos, dtype=float)
    vel = np.array(initial_vel, dtype=float)
    
    for _ in range(steps):
        positions.append(pos.copy())
        
        # Calculate gravitational acceleration
        ax, ay, az = hand.gravitational_field(pos[0], pos[1], pos[2])
        acc = np.array([ax, ay, az])
        
        # Update velocity and position (Leapfrog integration)
        vel += acc * dt
        pos += vel * dt
        
        # Check if particle has escaped or crashed
        distance = np.linalg.norm(pos)
        if distance > 10 * hand.hand_size or distance < 0.01 * hand.hand_size:
            break
    
    return np.array(positions)

def plot_gravitational_field(hand):
    """
    Create a 2D visualization of the gravitational field
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create coordinate grids
    x = np.linspace(-0.5, 0.5, 50) * hand.hand_size
    y = np.linspace(-0.8, 0.8, 60) * hand.hand_size
    X, Y = np.meshgrid(x, y)
    
    # Calculate potential field
    Z_potential = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z_potential[j, i] = hand.gravitational_potential(X[j, i], Y[j, i], 0)
    
    # Plot potential field
    contour1 = ax1.contour(X, Y, Z_potential, levels=20, colors='blue', alpha=0.7)
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.set_title(f'Gravitational Potential Field\n(Mass = {hand.total_mass:.2e} kg)')
    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Y position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot hand mass distribution
    palm_points = hand.mass_points[:20]  # First 20 are palm points
    finger_points = hand.mass_points[20:]  # Rest are finger points
    
    ax1.scatter(palm_points[:, 0], palm_points[:, 1], c='red', s=30, alpha=0.8, label='Palm')
    ax1.scatter(finger_points[:, 0], finger_points[:, 1], c='orange', s=20, alpha=0.8, label='Fingers')
    ax1.legend()
    
    # Calculate and plot gravitational field vectors
    skip = 5  # Skip points for cleaner visualization
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(0, len(x), skip):
        for j in range(0, len(y), skip):
            ax, ay, _ = hand.gravitational_field(X[j, i], Y[j, i], 0)
            U[j, i] = ax
            V[j, i] = ay
    
    # Normalize field vectors for visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = np.where(magnitude > 0, U / magnitude, 0)
    V_norm = np.where(magnitude > 0, V / magnitude, 0)
    
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               U_norm[::skip, ::skip], V_norm[::skip, ::skip], 
               magnitude[::skip, ::skip], cmap='viridis', alpha=0.7)
    
    ax2.scatter(palm_points[:, 0], palm_points[:, 1], c='red', s=30, alpha=0.8, label='Palm')
    ax2.scatter(finger_points[:, 0], finger_points[:, 1], c='orange', s=20, alpha=0.8, label='Fingers')
    ax2.set_title('Gravitational Field Vectors')
    ax2.set_xlabel('X position (m)')
    ax2.set_ylabel('Y position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_particle_orbits(hand):
    """
    Simulate and plot particle orbits around the massive hand
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hand mass distribution
    ax.scatter(hand.mass_points[:, 0], hand.mass_points[:, 1], hand.mass_points[:, 2], 
               c='red', s=50, alpha=0.8, label='Hand Mass')
    
    # Simulate multiple particle orbits with different initial conditions
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    
    for i, color in enumerate(colors):
        # Different initial positions and velocities
        angle = i * 2 * np.pi / len(colors)
        r = 0.5 * hand.hand_size
        
        initial_pos = [r * np.cos(angle), r * np.sin(angle), 0.1 * hand.hand_size]
        
        # Calculate approximate orbital velocity
        v_orbital = hand.orbital_velocity(r) * 0.8  # Slightly elliptical orbit
        initial_vel = [-v_orbital * np.sin(angle), v_orbital * np.cos(angle), 0]
        
        # Simulate orbit
        orbit = simulate_particle_orbit(hand, initial_pos, initial_vel, dt=0.01, steps=5000)
        
        if len(orbit) > 10:  # Only plot if orbit is reasonable
            ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 
                   color=color, alpha=0.7, linewidth=2, label=f'Particle {i+1}')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Particle Orbits Around Massive Hand\n(Mass = {hand.total_mass:.2e} kg)')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = hand.hand_size
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range/2, max_range/2)
    
    plt.tight_layout()
    plt.show()

def analyze_mass_regimes():
    """
    Analyze different mass regimes and their gravitational effects
    """
    masses = np.logspace(3, 35, 100)  # From 1 kg to 10^35 kg
    hand_size = 0.2  # 20 cm hand
    
    schwarzschild_radii = 2 * G * masses / c**2
    escape_velocities = np.sqrt(2 * G * masses / hand_size)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Schwarzschild radius vs mass
    ax1.loglog(masses, schwarzschild_radii, 'b-', linewidth=2)
    ax1.axhline(y=hand_size, color='r', linestyle='--', label='Hand size (20 cm)')
    ax1.axvline(x=M_sun, color='g', linestyle='--', label='Solar mass')
    ax1.set_xlabel('Mass (kg)')
    ax1.set_ylabel('Schwarzschild Radius (m)')
    ax1.set_title('Schwarzschild Radius vs Mass')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Escape velocity vs mass
    ax2.loglog(masses, escape_velocities, 'r-', linewidth=2)
    ax2.axhline(y=c, color='b', linestyle='--', label='Speed of light')
    ax2.set_xlabel('Mass (kg)')
    ax2.set_ylabel('Escape Velocity (m/s)')
    ax2.set_title('Escape Velocity from Hand Surface vs Mass')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gravitational acceleration at surface
    surface_gravity = G * masses / hand_size**2
    ax3.loglog(masses, surface_gravity, 'g-', linewidth=2)
    ax3.axhline(y=9.81, color='r', linestyle='--', label='Earth surface gravity')
    ax3.set_xlabel('Mass (kg)')
    ax3.set_ylabel('Surface Gravity (m/s²)')
    ax3.set_title('Surface Gravitational Acceleration vs Mass')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Find critical masses
    critical_mass_bh = hand_size * c**2 / (2 * G)  # When Schwarzschild radius = hand size
    critical_mass_light = hand_size * c**2 / (2 * G)  # When escape velocity = c
    
    print(f"Critical Analysis for 20cm Hand:")
    print(f"Mass for black hole formation: {critical_mass_bh:.2e} kg ({critical_mass_bh/M_sun:.2e} solar masses)")
    print(f"Mass for light-speed escape velocity: {critical_mass_light:.2e} kg")
    print(f"Comparison to Earth mass: {critical_mass_bh/(5.97e24):.2e} times Earth mass")

def main():
    """
    Main function to run all simulations and analyses
    """
    print("Massive Hand Gravitational Physics Simulation")
    print("=" * 50)
    
    # Test different mass regimes
    masses_to_test = [
        (1e6, "Low mass regime (1,000 kg)"),
        (1e24, "Intermediate mass regime (10^24 kg)"),
        (1e28, "High mass regime (10^28 kg)")
    ]
    
    for mass, description in masses_to_test:
        print(f"\n{description}")
        print("-" * 30)
        
        hand = MassiveHand(total_mass=mass, hand_size=0.2)
        
        print(f"Total mass: {hand.total_mass:.2e} kg")
        print(f"Schwarzschild radius: {hand.schwarzschild_radius():.2e} m")
        print(f"Escape velocity at surface: {hand.escape_velocity(hand.hand_size):.2e} m/s")
        print(f"Orbital velocity at 1m: {hand.orbital_velocity(1.0):.2e} m/s")
        
        # Visualizations for intermediate mass regime
        if mass == 1e24:
            print("\nGenerating visualizations...")
            plot_gravitational_field(hand)
            plot_particle_orbits(hand)
    
    # Analyze mass regimes
    print("\nAnalyzing mass regimes...")
    analyze_mass_regimes()
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
