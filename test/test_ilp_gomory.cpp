#include <iostream>
#include <cassert>
#include <string>
#include "../LP/Rational.h"
#include "../ILP/ILPGomoryCut.h"

// 轻量级测试宏
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
// 验证 Gomory 割平面法的“一刀切”能力
void test_pure_ilp_gomory() {
    std::cout << "\n>>> 运行测试 1: 纯整数规划 (Gomory Cut) <<<\n";
    ILPGomoryCut ilp;
    
    // 经典测试题：
    // Max Z = 5*x1 + 8*x2
    ilp.setObjective(ILPGomoryCut::MAX, {
        {"x1", Rational(5)}, {"x2", Rational(8)}
    });

    // 1. x1 + x2 <= 6
    ilp.addConstraint({{"x1", Rational(1)}, {"x2", Rational(1)}}, "<=", Rational(6));
    // 2. 5*x1 + 9*x2 <= 45
    ilp.addConstraint({{"x1", Rational(5)}, {"x2", Rational(9)}}, "<=", Rational(45));

    // 注意：标准的 Gomory 割平面法要求所有变量（包括底层的松弛变量）都必须是整数。
    // 由于我们这里的约束系数和右侧常数都是整数，松弛变量自然也是整数。
    
    // 开启 debug 模式，亲眼看看它切了几刀！
    Simplex::Status status = ilp.solve(true); 
    
    ASSERT_TRUE(status == Simplex::OPTIMAL);
    
    // LP 松弛的最优解是 x1=9/4, x2=15/4, Z=165/4 (41.25)
    // Gomory 切割后的纯整数最优解应该是 x1=0, x2=5, Z=40
    
    auto solution = ilp.getSolution();
    ASSERT_EQUAL_RATIONAL(solution["x1"], Rational(0));
    ASSERT_EQUAL_RATIONAL(solution["x2"], Rational(5));
    // ILPGomoryCut 的底层是借用的 LPModel 的解，我们需要手动算一下目标值验证
    Rational optimalZ = solution["x1"] * Rational(5) + solution["x2"] * Rational(8);
    ASSERT_EQUAL_RATIONAL(optimalZ, Rational(40));
    
    std::cout << "[PASS] Gomory Cut 纯整数规划测试通过！\n";
}

int main() {
    std::cout << "==============================================\n";
    std::cout << " ExactOR - ILP Gomory割平面法 自动化测试单元\n";
    std::cout << "==============================================\n";
    
    test_pure_ilp_gomory();
    
    std::cout << "\n==============================================\n";
    std::cout << "所有测试用例均已完美通过! (All tests passed.)\n";
    std::cout << "==============================================\n";
    
    return 0;
}