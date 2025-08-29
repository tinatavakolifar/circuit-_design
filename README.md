# Circuit Design Visualizer

## Overview

**Graph Analyser** is a Python tool for analyzing and visualizing **combinational** and **sequential circuits** based on CSV input tables.

It can:

- Minimize Boolean functions for small combinational circuits using the **Quine–McCluskey algorithm**.
- Detect **special circuits** like adders, subtractors, flip-flops, multiplexers, decoders, and registers.
- Generate **PDF visualizations**:
  - Combinational circuits: hazard-free and simplified versions using **Schemdraw**.
  - Sequential circuits: next-state and output logic diagrams.
  - **Note:** Sequential circuit state diagram visualization is currently **not implemented**.

---

## Features

- **Combinational circuits**:
  - Handles circuits up to 8 inputs using Q-M algorithm.
  - Detects half/full adders, subtractors, multipliers, comparators, decoders, encoders, multiplexers.
  - **Note:** Circuits with more than 8 inputs currently use placeholders for **Espresso algorithm** (future work).
  - Visualization implemented via **Schemdraw**.

- **Sequential circuits**:
  - Generates combinational logic diagrams for next-state and output functions.
  - **Sequential state diagram visualization is currently not implemented.**
  - Supports small circuits (≤5 inputs) for exact minimization; placeholders exist for larger circuits (BDD/State encoding, heuristic methods).
  - Visualization limited to next-state and output logic; state diagrams are not generated.

- **Organized output**:
  - All PDF diagrams are saved in a dedicated `output_pdfs` folder.

- **CSV input**:
  - Combinational: truth table (`input, output`)
  - Sequential: state table (`Qt, input, Qt+1, output`)
  - **Access CSV files:** Include them in the `examples/` folder and reference in README links. Users can download or clone the repo to use them.

---

## Installation

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd circuit_design
```

Install dependencies:

Python ≥ 3.9

Schemdraw Python package:

```bash
pip install schemdraw
```

---

## Usage

Prepare a CSV input file:

Combinational example – Half Adder (examples/half_adder.csv):

```csv
input,output
A,B,Cout,S
0,0,0,0
0,1,0,1
1,0,0,1
1,1,1,0
```

Sequential example – JK Flip-Flop (examples/jk_ff.csv):

```csv
qt,input,qt+1,output
Q,J,K,Qnext
0,0,0,0,0
0,0,1,0,0
0,1,0,1,1
0,1,1,1,0
1,0,0,1,1
1,0,1,1,0
1,1,0,0,1
1,1,1,0,0
```

Run the program:

```bash
python circuit_design.py
```

Follow prompts to enter the CSV file path.

PDFs will be generated automatically in `output_pdfs`.

---

## Output

`output_pdfs/` folder contains:

**Combinational circuits:**

- `*_hazard_free.pdf` – all prime implicants
- `*_simplified.pdf` – essential prime implicants only

**Sequential circuits:**

- `*_nextstate_<bit>.pdf` – next-state logic for each state bit
- `*_output_<bit>.pdf` – output logic for each output bit
- **Note:** `state_diagram.pdf` is currently **not generated** as state diagram visualization is not implemented.

---

## Screenshots

Half Adder – Simplified

---

## Notes

- Current limitations:
  - Combinational circuits with >8 inputs use placeholder for Espresso algorithm.
  - Combinational circuit visualization is implemented via **Schemdraw**.
  - Sequential circuits with >5 inputs use placeholder methods for BDD/State encoding and heuristic minimization.
  - Sequential state diagram visualization is **not implemented**.

---

## Future improvements

- GUI interface for easier interaction.
- Implement large circuit minimization algorithms.
- Add more automatic detection of special sequential circuits.
- Implement sequential circuit state diagram visualization.

---

## Example CSVs

Included in the `examples/` folder:

- `half_adder.csv` – combinational
- `jk_ff.csv` – sequential
- `multiplexer.csv` – combinational

---

## Credits

Author: Tina Tavakolifar  
Uses Schemdraw for circuit visualization.
