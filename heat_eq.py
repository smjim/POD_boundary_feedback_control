import numpy as np
import matplotlib.pyplot as plt
import time 

# Define the 1D Heat Equation Solver
class HeatEquationSolver1D:
	def __init__(self, nx, nt_snapshots, L, T, alpha, method='Euler', nt=1000):
		"""
		Initialize the solver for the 1D heat equation.
		
		Parameters:
		- nx: Number of spatial grid points
		- nt_snapshots: Number of POD-analyzable snapshots to be saved 
		- L: Length of the spatial domain
		- T: Total time of simulation
		- alpha: Thermal diffusivity
		- method: Numerical method for time-stepping ('Euler' or 'RK4')
		- nt: Number of time steps
		"""
		self.nx = nx
		self.nt_snapshots = nt_snapshots
		self.dx = L / (nx - 1)
		self.dt = T / nt
		self.alpha = alpha
		self.nt = nt
		self.method = method
		self.u = np.zeros(nx)  # Initialize temperature field
		self.x = np.linspace(0, L, nx)  # Spatial grid points
		
		# Initial condition:

		# Gaussian profile
		#self.u[:] = np.exp(-10 * (self.x - L / 2)**2)
		# Linear profile
		self.u[:] = 2 * (self.x - L / 2)
		# Constant profile
		#self.u[:] = 2 
	
	def euler_step(self):
		"""Euler method for time-stepping."""
		u_new = np.copy(self.u)
		for i in range(1, self.nx - 1):
			u_new[i] = self.u[i] + self.alpha * self.dt / self.dx**2 * (self.u[i+1] - 2*self.u[i] + self.u[i-1])

		# Boundary conditions
		u_new[0] = 0
		u_new[-1] = 0

		self.u = u_new
	
	def rk4_step(self):
		"""Runge-Kutta 4th order method for time-stepping."""
		def heat_rhs(u):
			# Right-hand side of the heat equation
			du_dt = np.zeros_like(u)
			for i in range(1, self.nx - 1):
				du_dt[i] = self.alpha * (u[i+1] - 2*u[i] + u[i-1]) / self.dx**2
			return du_dt
		
		k1 = heat_rhs(self.u)
		k2 = heat_rhs(self.u + 0.5 * self.dt * k1)
		k3 = heat_rhs(self.u + 0.5 * self.dt * k2)
		k4 = heat_rhs(self.u + self.dt * k3)
		
		# Boundary conditions
		self.u[0] = 0
		self.u[-1] = 0

		self.u += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
	
	def run_simulation(self):
		"""Run the simulation using the selected numerical method."""
		snapshots = np.zeros((self.nx, self.nt_snapshots)) # nt_snapshots = number of snapshots desired
		snapshot_interval = self.nt // self.nt_snapshots

		print(f'Running {self.method} Method:')
		start_time = time.time()
		for t in range(self.nt):
			if self.method == 'Euler':
				self.euler_step()
			elif self.method == 'RK4':
				self.rk4_step()
			if t % snapshot_interval == 0:
				snapshots[:, t // snapshot_interval] = self.u
				print(f'\tSimulation progress: {t} frames, {100*t/self.nt:.2f} % complete')
		np.save('snapshots.npy', snapshots)
		duration = time.time() - start_time
		print(f'{self.method} Method duration: {duration:4f} s, {duration/self.nt:.4e} s/ iteration')
	
	def plot_solution(self):
		"""Plot the final temperature distribution."""
		plt.plot(self.x, self.u, label=f'{self.method} Method')
		plt.title(f'Temperature Distribution ({self.method})')
		plt.xlabel('Position (x)')
		plt.ylabel('Temperature (u)')
		plt.legend()
		plt.grid(True)
		plt.show()


# Parameters for the simulation
nx = 100		 	# Number of spatial points
L = 1.0			  	# Length of the rod
T = 1.0			  	# Total time for simulation
alpha = 0.01	 	# Thermal diffusivity
nt = int(5e4)	 	# Number of time steps
nt_snapshots = 50	# Number of POD-analyzable snapshots

# Run simulation with Euler method
solver_euler = HeatEquationSolver1D(nx, nt_snapshots, L, T, alpha, method='Euler', nt=nt)
solver_euler.run_simulation()
solver_euler.plot_solution()

# Run simulation with RK4 method
#solver_rk4 = HeatEquationSolver1D(nx, nt_snapshots, L, T, alpha, method='RK4', nt=nt)
#solver_rk4.run_simulation()
#solver_rk4.plot_solution()

