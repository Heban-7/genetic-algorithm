# Genetic Algorithm Implementation in MeTTa and Python

A dual implementation of a simple Genetic Algorithm (GA) system that evolves binary strings to match a target pattern. This project demonstrates the core components of genetic algorithms implemented in both **MeTTa** (a declarative knowledge representation language) and **Python**.

## Task

Build the core components of a simple GA system in MeTTa and Python:

1. **Solution Representation**: Represent each candidate solution as a binary string (length of 10) using a list
2. **Scoring**: Define and implement a simple fitness function that can evaluate the quality of solutions (match a target binary string)
3. **Selection**: Implement a selection method: roulette wheel selection
4. **Variation Operations**: Implement crossover and mutation operations
   - **Crossover**: Single-point crossover to swap parts of two parents
   - **Mutation**: Flip each bit with a small probability

## Project Structure

```
genetic-algorithm/
├── src/
│   ├── Python_Genetic_Algorithm.py    # Python implementation
│   └── MeTTa_Genetic_Algorithm.metta  # MeTTa implementation
├── tutorial.metta                     # MeTTa language tutorial
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🚀 Features

### Core GA Components Implemented

- **Binary String Representation**: 10-bit chromosomes represented as lists
- **Fitness Function**: Counts matching bits between candidate and target
- **Roulette Wheel Selection**: Probability-based parent selection
- **Single-Point Crossover**: Swaps genetic material between parents
- **Bit-Flip Mutation**: Random bit flipping with configurable rate
- **Population Evolution**: Generational replacement with elitism

### Configuration Parameters

- **Population Size**: 100 individuals
- **Mutation Rate**: 0.1 (10% chance per bit)
- **Crossover Rate**: 0.95 (95% chance per pair)
- **Target String**: Configurable 10-bit binary string
- **Max Generations**: 500 (with early termination on solution)

## 📋 Requirements

### Python Implementation
```bash
python3
```

### MeTTa Implementation
```bash
hyperon
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd genetic-algorithm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Python Implementation

Run the Python genetic algorithm:

```bash
cd src
python Python_Genetic_Algorithm.py
```


### MeTTa Implementation

Run the MeTTa genetic algorithm:

```bash
cd src
metta MeTTa_Genetic_Algorithm.metta
```

## Algorithm Overview

1. **Initialization**: Create random population of binary strings
2. **Evaluation**: Calculate fitness for each individual
3. **Selection**: Choose parents using roulette wheel selection
4. **Crossover**: Create offspring using single-point crossover
5. **Mutation**: Apply bit-flip mutation to offspring
6. **Replacement**: Replace population with new generation
7. **Termination**: Stop when target is found or max generations reached

## Learning Resources

- **MeTTa Tutorial**: See `tutorial.metta` for MeTTa language basics
- **Genetic Algorithm Theory**: Understanding evolutionary computation principles
- **Hyperon Documentation**: For advanced MeTTa features

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the implementation.
