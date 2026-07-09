LPModel API Reference Manual

LPModel is a lightweight, high-precision Linear Programming solver modeling layer written in C++. Under the hood, it relies on a custom Rational class to completely eliminate floating-point precision errors. It features a built-in Two-Phase Simplex Method engine that automatically handles slack, surplus, and artificial variables.

1. Core Features

Lossless Precision: The entire pipeline uses a custom Rational fraction class for calculations, bidding farewell to rounding errors like 0.3333333.

User-Friendly: Variable names are mapped using std::string. The underlying engine automatically manages the scheduling and conversion between string names and matrix column indices (registers).

Comprehensive Constraint Support: Natively supports <=, >=, and = constraints, automatically inferring and generating the matrices required for the Big-M / Two-Phase method.

Silent and Compact: No redundant terminal outputs, making it highly suitable as a Model Context Protocol (MCP) tool or for integration into large-scale engineering projects.

2. Core Data Structures

2.1 Enums and States

LPModel::ObjSense

MAX: Maximize the objective function.

MIN: Minimize the objective function.

Simplex::Status (Returned by solve())

OPTIMAL: An optimal solution was successfully found.

UNBOUNDED: The problem is unbounded (divergent).

INFEASIBLE: No feasible solution exists (constraints are contradictory).

3. API Details

3.1 Initialization and Objective Function Setup

LPModel();


Description: Constructs an empty linear programming model.

void setObjective(ObjSense s, const std::map<std::string, Rational>& terms);


Description: Sets the objective function.

Parameters:

s: Seek the maximum value (LPModel::MAX) or minimum value (LPModel::MIN).

terms: The polynomial of the objective function, where the key is the variable name (string) and the value is the coefficient (Rational). Variables not present in the map are assumed to have a coefficient of 0.

3.2 Adding Constraints

void addConstraint(std::map<std::string, Rational> terms, std::string op, Rational rhs);


Description: Adds a constraint to the model.

Parameters:

terms: The polynomial coefficient map on the left side of the constraint equation.

op: The operator, supporting only "<=", ">=", and "=".

rhs: The constant term on the right side of the constraint equation (negative values are handled automatically).

3.3 Solving and Retrieving Results

Simplex::Status solve();


Description: Compiles the current model and executes the Simplex method (automatically running the Two-Phase method if necessary).

Returns: The solving status enumeration.

Note: This method must be called before invoking any result retrieval interfaces.

Rational getObjectiveValue() const;


Description: Gets the objective function value Z under the optimal solution.

Prerequisite: solve() returned OPTIMAL.

std::map<std::string, Rational> getSolution() const;


Description: Gets the specific values of all registered variables under the optimal solution.

Returns: A dictionary mapping variable names to their optimal values.

Prerequisite: solve() returned OPTIMAL.

4. Quick Start Example

Below is an example of solving a complex linear programming problem that mixes equality, inequality, and negative constant terms.

Mathematical Model:

Objective Function: Min Z = 4*x1 + x2

Constraints:

3*x1 + x2 = 3

-4*x1 - 3*x2 <= -6

x1 + 2*x2 <= 4

C++ Calling Code:
```cpp
#include <iostream>
#include "Rational.h"
#include "Simplex.h"
#include "LPModel.h"

int main() {
    LPModel model;

    // 1. Set the objective function (Minimize)
    model.setObjective(LPModel::MIN, {
        {"x1", Rational(4)}, 
        {"x2", Rational(1)}
    });

    // 2. Add constraints
    // Constraint 1: 3*x1 + x2 = 3
    model.addConstraint({{"x1", Rational(3)}, {"x2", Rational(1)}}, "=", Rational(3));
    
    // Constraint 2: -4*x1 - 3*x2 <= -6
    model.addConstraint({{"x1", Rational(-4)}, {"x2", Rational(-3)}}, "<=", Rational(-6));
    
    // Constraint 3: x1 + 2*x2 <= 4
    model.addConstraint({{"x1", Rational(1)}, {"x2", Rational(2)}}, "<=", Rational(4));

    // 3. Execute the solver
    Simplex::Status status = model.solve();

    // 4. Handle the results
    if (status == Simplex::OPTIMAL) {
        std::cout << "Successfully solved!\n";
        std::cout << "Optimal Objective Value Z = " << model.getObjectiveValue() << "\n";
        
        auto solution = model.getSolution();
        for (const auto& pair : solution) {
            std::cout << "Variable " << pair.first << " = " << pair.second << "\n";
        }
    } else if (status == Simplex::UNBOUNDED) {
        std::cout << "The problem is unbounded!\n";
    } else if (status == Simplex::INFEASIBLE) {
        std::cout << "No feasible solution!\n";
    }

    return 0;
}
```