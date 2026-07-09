#include<iostream>
#include<cassert>
#include"../Graph/KMAlgorithm.h"
#define ASSERT_TRUE(condition) do{if(!(condition)){std::cerr<<"[FAIL] Assertion failed: "<<#condition<<" at line "<<__LINE__<<"\n";exit(1);}}while(false)
#define ASSERT_EQUAL(a,b) do{if((a)!=(b)){std::cerr<<"[FAIL] Assertion failed: "<<(a)<<" != "<<(b)<<" at line "<<__LINE__<<"\n";exit(1);}}while(false)
void test_delivery_assignment(){
    std::cout<<"\n>>> 运行测试: 经典外卖派单冲突 (KM 算法) <<<\n";
    KMAlgorithm km;
    km.addEdge("张三","单1",5);
    km.addEdge("张三","单2",4);
    km.addEdge("张三","单3",2);
    km.addEdge("李四","单1",5);
    km.addEdge("李四","单2",3);
    km.addEdge("李四","单3",2);
    km.addEdge("王五","单1",2);
    km.addEdge("王五","单2",2);
    km.addEdge("王五","单3",4);
    long long maxScore=km.solve(true);
    ASSERT_EQUAL(maxScore,13);
    auto matches=km.getMatches();
    std::cout<<"\n[求解成功!] 找到全局最高收益分配方案: \n";
    for(const auto& pair:matches){
        std::cout<<"外卖员 ["<<pair.first<<"] -> 分配 -> ["<<pair.second<<"]\n";
    }
    ASSERT_TRUE(matches["李四"]=="单1");
    ASSERT_TRUE(matches["张三"]=="单2");
    ASSERT_TRUE(matches["王五"]=="单3");
    std::cout<<"\n[PASS] KM 算法测试通过！总效率分: "<<maxScore<<"\n";
}
int main(){
    std::cout<<"==============================================\n";
    std::cout<<" ExactOR - KM 图论指派算法 自动化测试单元\n";
    std::cout<<"==============================================\n";
    test_delivery_assignment();
    std::cout<<"==============================================\n";
    std::cout<<"所有测试用例均已完美通过! (All tests passed.)\n";
    std::cout<<"==============================================\n";
    return 0;
}