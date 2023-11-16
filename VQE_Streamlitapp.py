import streamlit as st
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("Variational Quantum Eigensolver (VQE) Tool")

# Sidebar for optimizer selection
st.sidebar.header("Optimizer Settings")
optimizer_choice = st.sidebar.selectbox("Select Optimizer", ["Gradient Descent", "Adam"])
stepsize = st.sidebar.number_input("Step Size", min_value=0.01, max_value=1.0, value=0.4)

# Function to get optimizer
def get_optimizer():
    if optimizer_choice == "Gradient Descent":
        return qml.GradientDescentOptimizer(stepsize)
    elif optimizer_choice == "Adam":
        return qml.AdamOptimizer(stepsize)

# Hamiltonian design
st.header("Hamiltonian Design")
symbols_input = st.text_input("Enter atomic symbols (comma separated)", "H, H")
coordinates_input = st.text_area("Enter coordinates (as rows of 3 values)", "0.0, 0.0, -0.6614\n0.0, 0.0, 0.6614")

# Parse Hamiltonian inputs
def parse_hamiltonian_input(symbols_input, coordinates_input):
    # Trim spaces from symbols and split by comma
    symbols = [symbol.strip() for symbol in symbols_input.split(",")]

    # Process coordinates; split by newlines and then by commas
    coordinates_list = []
    for line in coordinates_input.split("\n"):
        coordinates_list.extend([float(coord) for coord in line.split(",") if coord.strip()])
    coordinates = np.array(coordinates_list).reshape(-1, 3)

    return symbols, coordinates

# VQE Function
def run_vqe(symbols, coordinates):
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    dev = qml.device("default.qubit", wires=qubits)
    electrons = 2  # This could be made dynamic based on user input

    hf_state = qml.qchem.hf_state(electrons, qubits)

    def circuit(param, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(qubits))
        return qml.expval(H)

    opt = get_optimizer()
    theta = np.array(0.0, requires_grad=True)
    max_iterations = 100
    conv_tol = 1e-06

    energy = [cost_fn(theta)]
    angle = [theta]

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        energy.append(cost_fn(theta))
        angle.append(theta)
        conv = np.abs(energy[-1] - prev_energy)

        if conv <= conv_tol:
            break

    return energy, angle, n

# Button to run VQE
if st.button("Run VQE"):
    symbols, coordinates = parse_hamiltonian_input(symbols_input, coordinates_input)
    energy, angle, n = run_vqe(symbols, coordinates)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(energy, 'go-', label="Energy")
    axs[0].set_title("Energy Convergence")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Energy")

    axs[1].plot(angle, 'ro-', label="Angle")
    axs[1].set_title("Angle Convergence")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Angle")

    st.pyplot(fig)
