#include <iostream>
#include <cassert>
#include <string>
#include <cmath>
#include "../LP/Rational.h"
#include "../ILP/ILPBranchAndBound.h"

// 轻量级测试宏，用于验证结果
#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "[FAIL] Assertion failed: " << #condition << " at line " << __LINE__ << "\n"; \
            exit(1); \
        } \
    } while (false)

#define ASSERT_EQUAL_RATIONAL(a, b) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "[FAIL] Assertion failed: " << (a) << " != " << (b) << " at line " << __LINE__ << "\n"; \
            exit(1); \
        } \
    } while (false)


// 测试用例 1: 纯整数规划 (Pure ILP)
// 验证常规的分支定界求解能力
void test_pure_ilp() {
    std::cout << "\n>>> 运行测试 1: 纯整数规划 (Pure ILP) <<<\n";
    ILPBranchAndBound ilp;
    
    // Max Z = 5*x1 + 8*x2
    ilp.setObjective(ILPBranchAndBound::MAX, {
        {"x1", Rational(5)}, {"x2", Rational(8)}
    });

    // 1. x1 + x2 <= 6
    ilp.addConstraint({{"x1", Rational(1)}, {"x2", Rational(1)}}, "<=", Rational(6));
    // 2. 5*x1 + 9*x2 <= 45
    ilp.addConstraint({{"x1", Rational(5)}, {"x2", Rational(9)}}, "<=", Rational(45));

    ilp.addIntegerVariable("x1");
    ilp.addIntegerVariable("x2");

    Simplex::Status status = ilp.solve(true); // 开启调试模式追踪搜索树
    
    ASSERT_TRUE(status == Simplex::OPTIMAL);
    
    // LP 松弛的最优解是 x1=9/4, x2=15/4, Z=165/4 (41.25)
    // ILP 寻优后的纯整数最优解应该是 x1=0, x2=5, Z=40
    ASSERT_EQUAL_RATIONAL(ilp.getObjectiveValue(), Rational(40));
    
    auto solution = ilp.getSolution();
    ASSERT_EQUAL_RATIONAL(solution["x1"], Rational(0));
    ASSERT_EQUAL_RATIONAL(solution["x2"], Rational(5));
    
    std::cout << "[PASS] 测试 1 通过\n";
}

// 测试用例 2: 混合整数规划 (MILP)
// 验证只约束部分变量为整数时的求解正确性
void test_mixed_ilp() {
    std::cout << "\n>>> 运行测试 2: 混合整数规划 (MILP) <<<\n";
    ILPBranchAndBound ilp;
    
    // Max Z = x1 + 2*x2
    ilp.setObjective(ILPBranchAndBound::MAX, {
        {"x1", Rational(1)}, {"x2", Rational(2)}
    });

    // 2*x1 + 3*x2 <= 8
    ilp.addConstraint({{"x1", Rational(2)}, {"x2", Rational(3)}}, "<=", Rational(8));

    // 仅要求 x2 为整数，x1 可以是小数
    ilp.addIntegerVariable("x2");

    Simplex::Status status = ilp.solve(true); // 开启调试模式追踪搜索树
    ASSERT_TRUE(status == Simplex::OPTIMAL);

    // 分析：
    // 如果 x2=2 (整数), 则 2*x1 + 6 <= 8 => x1 <= 1. 此时 Z = 1 + 2*2 = 5
    // 如果 x2=3 (超出), 违背非负
    // 最优解应该是 x1=1, x2=2
    auto solution = ilp.getSolution();
    ASSERT_EQUAL_RATIONAL(ilp.getObjectiveValue(), Rational(5));
    ASSERT_EQUAL_RATIONAL(solution["x2"], Rational(2)); // 必须是整数
    ASSERT_EQUAL_RATIONAL(solution["x1"], Rational(1)); 
    
    std::cout << "[PASS] 测试 2 通过\n";
}

// 测试用例 3: 连续域有解但整数域无解 (Infeasible ILP)
void test_infeasible_ilp() {
    std::cout << "\n>>> 运行测试 3: 无可行整数解 (Infeasible ILP) <<<\n";
    ILPBranchAndBound ilp;
    
    // Max Z = x1
    ilp.setObjective(ILPBranchAndBound::MAX, {{"x1", Rational(1)}});

    // 2*x1 = 3  (唯一连续解是 x1 = 1.5)
    ilp.addConstraint({{"x1", Rational(2)}}, "=", Rational(3));

    ilp.addIntegerVariable("x1");

    Simplex::Status status = ilp.solve(true); // 开启调试模式追踪搜索树
    
    // LP 能够解出 1.5，但是 ILP 要求必须是整数，所以最终返回的应该是 INFEASIBLE
    ASSERT_TRUE(status == Simplex::INFEASIBLE);
    
    std::cout << "[PASS] 测试 3 通过\n";
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "   ExactOR - ILP 分支定界法 自动化测试单元\n";
    std::cout << "==========================================\n";
    
    test_pure_ilp();
    test_mixed_ilp();
    test_infeasible_ilp();
    
    std::cout << "\n==========================================\n";
    std::cout << "所有测试用例均已完美通过! (All tests passed.)\n";
    std::cout << "==========================================\n";
    
    return 0;
}