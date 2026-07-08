#include <iostream>
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
