# Coupled Learning Architecture for Mechanical Networks  
*Design Rationale, Abstractions, and Naming Conventions*

---

## 1. Motivation

As coupled learning systems scale from minimal toy examples to large mechanical or physical networks (e.g., spring networks, membrane patch networks, or hardware-in-the-loop systems), architectural clarity becomes critical. Without a strict separation between **task definition**, **physics execution**, **learning rules**, and **visualization**, implementations frequently accumulate ambiguous variables, mixed responsibilities, and experiment-specific assumptions that obscure physical meaning and inhibit reuse.

This document defines a **backend-agnostic, physics-consistent architecture** for coupled learning systems. Particular emphasis is placed on the distinction between **learnable parameters** and **measured physical states**, as confusion between these quantities is a common source of silent implementation errors.

---

## 2. Conceptual Decomposition

We decompose the system into four conceptual layers.

### 2.1 Problem (Task Definition)

The **Problem** defines *what* the system should achieve, independently of the underlying physical realization.

Responsibilities:
- Specify **inputs** (sources) and **targets** (outputs)
- Define how **clamped boundary conditions** are constructed from the free state (e.g., nudging rules)
- Optionally define **evaluation metrics** (error, symmetry violation, etc.)

The Problem:
- Does **not** solve physics
- Does **not** update parameters
- May depend on free-state measurements to construct clamped constraints

---

### 2.2 Backend (Physics Execution)

The **Backend** defines *how* the physical system responds to parameters and boundary conditions.

Responsibilities:
- Store and manage **learnable parameters**
- Define system **topology and indexing** (nodes, edges, patches)
- Execute **free** and **clamped** equilibrium solves
- Extract **measured physical states**

Examples:
- Pure Python spring network backend
- COMSOL membrane backend
- Hardware backend (sensors + actuators)

The Backend:
- Does **not** define learning rules
- Does **not** define objectives
- Must faithfully implement physical laws

---

### 2.3 Learner (Update Rules)

The **Learner** defines *how* parameters are updated from physical contrast signals.

Responsibilities:
- Implement **local update rules** (continuous or quantized)
- Enforce **parameter constraints** (bounds, discretization)
- Map measured contrasts to parameter updates

The Learner:
- Does **not** execute physics
- Does **not** construct boundary conditions

---

## 3. Engine (Orchestration)

The **Engine** coordinates the coupled learning loop:

1. Run free equilibrium
2. Construct clamped boundary conditions
3. Run clamped equilibrium
4. Compute local contrasts
5. Update parameters
6. Log results

The Engine should remain intentionally simple and stable as the system scales.

---

## 4. Learnable Parameters vs Measured States (Critical Distinction)

A recurring source of implementation error in mechanical coupled learning systems is the conflation of **learnable parameters** with **measured physical quantities**. These must be treated as fundamentally distinct objects.

### 4.1 Learnable Parameters (Slow Variables)

Learnable parameters are updated **between** learning iterations.

Examples:
- Spring rest length
- Spring stiffness
- Membrane patch stiffness
- Conductance in flow networks

Properties:
- Stored explicitly
- Indexed per element (edge, patch, etc.)
- Persist across equilibrium solves
- Updated only by the Learner

---

### 4.2 Measured States (Fast Variables)

Measured states are computed **within** an equilibrium solve.

Examples:
- Node positions
- Geometric edge lengths
- Membrane height or displacement
- Pressure, voltage, flow

Properties:
- Derived from physics
- Do **not** persist across iterations
- Never directly updated by learning

---

## 5. Spring Networks: Correct Physical Model

### 5.1 Edge Characterization

Each edge \( e \) connecting nodes \( i \) and \( j \) is characterized by:

- **Learnable parameters**
  - Rest length \( \ell_e \)
  - Stiffness \( k_e \) (fixed or learnable)

- **Measured state**
  - Geometric length  
    $$L_e(\mathbf{x}) = \|\mathbf{x}_i - \mathbf{x}_j\|$$

- **Extension / strain proxy**
  $$s_e = L_e(\mathbf{x}) - \ell_e$$

---

### 5.2 Energy and Forces (Non-Negotiable)

The physical model must explicitly include rest length:

$$
E(\mathbf{x}; \ell, k)
= \sum_e \frac{1}{2} k_e \left(L_e(\mathbf{x}) - \ell_e\right)^2
$$

Forces on nodes are computed as gradients of this energy with respect to node positions.

> Any implementation that computes forces or energy using only geometric edge lengths and ignores rest length is physically incorrect.

---

## 6. Common Failure Modes (Observed in Practice)

### 6.1 Rest Length Ignored in Physics

A frequent silent failure:
- Rest length is initialized
- Rest length is updated
- Rest length is **never used** in the solver

Result: learning appears to “work” numerically but has no physical meaning.

---

### 6.2 Edge Index Used as Length

Another common error:
- Edge index (0,1,2,…) is implicitly treated as a length
- Occurs when arrays are misused or improperly broadcast

This can produce smooth-looking plots that are physically meaningless.

---

### 6.3 Measured Length Confused with Rest Length

Using the same variable name (e.g., `L`) for:
- Learnable rest length
- Measured geometric length

This leads to accidental overwriting and incorrect updates.

---

## 7. Enforced Invariants and Guardrails

To prevent these failures, the following invariants must be enforced.

### 7.1 Shape and Type Invariants

- `rest_length.shape == (E,)`
- `stiffness.shape == (E,)`
- `edge_length.shape == (E,)`
- `node_positions.shape == (N, dim)`

---

### 7.2 Energy Sensitivity Check

At fixed node positions \( \mathbf{x} \):

$$
E(\mathbf{x}; \ell + \varepsilon) \neq E(\mathbf{x}; \ell)
$$

If this condition is violated, rest length is not influencing the physics.

---

### 7.3 Measurement Sanity Check

- Edge length must be computed from node coordinates
- Edge length must not equal edge indices or constant arrays

---

## 8. Logging Requirements (Unambiguous Separation)

Each run must log **both** learnable parameters and measured states.

Required histories:
- `rest_length_history` — shape (T, E)
- `edge_length_free_history` — shape (T, E)
- `edge_length_clamped_history` — shape (T, E)

Optional:
- symmetry error
- cost or contrast

> Never overwrite rest lengths with measured lengths.

---

## 9. Architectural Principle (Summary)

> **Learning updates parameters. Physics updates states.**  
> If a quantity changes during equilibrium, it is a measured state.  
> If it changes between equilibria, it is a learnable parameter.

Failure to preserve this separation is the primary cause of misleading “successful” coupled learning simulations.

---

## 10. Intended Scope

This architecture supports:
- Mechanical spring network replication
- Membrane-based coupled learning
- Simulation-to-hardware transitions
- Scalable experimentation without structural rewrites

It is designed to remain valid as system size, complexity, and physical fidelity increase.

## Usage Notes for New Implementations

This document serves as a **non-negotiable specification** for all coupled learning implementations in this project.

When starting a new experiment, backend, or learning rule, do not do below unless with author approval:
- redefine core abstractions (Problem, Backend, Learner, Engine).
- rename learnable parameters or measured states in ways that violate Section 4.
- introduce physics shortcuts that bypass Section 5.

Any new code, simulation, or experiment should be explainable using the terminology and structure defined here. If it cannot be explained within this framework, the design must be reconsidered.