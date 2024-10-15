import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

class PODAnalysis:
	def __init__(self, snapshots):
		"""
		Initialize the POD analysis with the set of snapshots.
		
		Parameters:
		- snapshots: 2D array of system states (spatial x time). Each column is a snapshot at a different time.
		"""
		self.snapshots = snapshots
		self.modes = None
		self.singular_values = None
		self.reduced_basis = None
	
	def compute_pod(self):
		"""
		Perform POD analysis on the snapshots using Singular Value Decomposition (SVD).
		"""
		# Perform SVD on the snapshot matrix
		U, S, Vt = svd(self.snapshots, full_matrices=False)
		
		# Store the results
		self.modes = U  # Left singular vectors (POD modes)
		self.singular_values = S  # Singular values
		self.reduced_basis = Vt.T  # Right singular vectors (time coefficients)
		
		# Explained energy by each mode (proportional to singular values)
		energy = (S**2) / np.sum(S**2)
		return energy
	
	def plot_modes(self, num_modes=3):
		"""
		Plot the first few POD modes.
		
		Parameters:
		- num_modes: Number of modes to plot
		"""
		for i in range(min(num_modes, self.modes.shape[1])):
			plt.plot(self.modes[:, i], label=f'Mode {i+1}')
		
		plt.title(f'First {num_modes} POD Modes')
		plt.xlabel('Spatial Grid Points')
		plt.ylabel('Mode Amplitude')
		plt.legend()
		plt.grid(True)
		plt.show()

	def plot_energy_distribution(self):
		"""
		Plot the distribution of energy captured by each mode.
		"""
		energy = (self.singular_values**2) / np.sum(self.singular_values**2)
		plt.plot(np.cumsum(energy), 'o-', label='Cumulative Energy')
		plt.title('Energy Distribution by POD Modes')
		plt.xlabel('Mode Number')
		plt.ylabel('Cumulative Energy')
		plt.grid(True)
		plt.legend()
		plt.show()

def plot_snapshots(snapshots, plot_individual_snapshots=True):
	"""
	Plots the evolution of the snapshots for the heat equation simulation.

	Parameters:
	- snapshots: numpy array of shape (nx, nt_snapshots), the snapshots data.
	- plot_individual_snapshots: bool, if True, also plot individual snapshots as lines.
	"""
	# Check the number of spatial points (nx) and snapshots in time (nt_snapshots)
	nx, nt_snapshots = snapshots.shape

	# Create spatial and time vectors
	x = np.linspace(0, 1, nx)  # Assuming spatial domain is [0, 1]
	time_snapshots = np.linspace(0, 1, nt_snapshots)  # Normalized time for plotting

	# Plot the heatmap of snapshot evolution
	plt.figure(figsize=(8, 6))
	plt.imshow(snapshots, aspect='auto', cmap='hot', extent=[0, 1, 0, 1], origin='lower')
	plt.colorbar(label='Temperature')
	plt.xlabel('Space (x)')
	plt.ylabel('Time (normalized)')
	plt.title('Evolution of the Heat Equation Snapshots')
	plt.show()

	# Optionally, plot individual snapshots
	if plot_individual_snapshots:
		plt.figure(figsize=(10, 6))
		for i in range(0, nt_snapshots, max(1, nt_snapshots // 10)):  # Plot 10 evenly spaced snapshots
			plt.plot(x, snapshots[:, i], label=f'Snapshot {i}')
		plt.xlabel('Space (x)')
		plt.ylabel('Temperature')
		plt.title('Snapshots of the Heat Equation Over Time')
		plt.legend(loc='upper right')
		plt.show()


# Sample code for integrating with heat_eq.py
# This assumes that you've already run the heat equation simulation and saved snapshots as a 2D array

if __name__ == "__main__":
	# Load snapshots from heat_eq.py (spatial x time snapshots)
	# For example, if snapshots were saved into a file 'snapshots.npy'
	snapshots = np.load('snapshots.npy')  # Shape (nx, nt)

	# Plot snapshot evolution
	plot_snapshots(snapshots)

	# Perform POD analysis on the snapshots
	pod = PODAnalysis(snapshots)
	energy = pod.compute_pod()
	
	# Plot the first 3 modes
	pod.plot_modes(num_modes=3)
	
	# Plot the energy distribution
	pod.plot_energy_distribution()

	# Print the explained energy by the first few modes
	print(f"Explained Energy by Modes 1-3: {energy[:3]}")

