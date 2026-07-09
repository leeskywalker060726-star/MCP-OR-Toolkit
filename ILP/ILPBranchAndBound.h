#ifndef ILP_BRANCH_AND_BOUND_H
#define ILP_BRANCH_AND_BOUND_H
#include<string>
#include<vector>
#include<map>
#include<set>
#include<stack>
#include<stdexcept>
#include"../LP/Rational.h"
#include"../LP/LPModel.h"
class ILPBranchAndBound{
public:
    enum ObjSense{MAX,MIN};
private:
    ObjSense sense;
    std::map<std::string,Rational> objTerms;
    struct ConstraintDef{
        std::map<std::string,Rational> terms;
        std::string op;
        Rational rhs;
    };
    std::vector<ConstraintDef> originalConstraints;
    std::set<std::string> integerVariables;
    Rational bestObjective;
    std::map<std::string,Rational> bestSolution;
    bool hasFeasibleIntegerSolution;
    struct SearchNode{
        std::vector<ConstraintDef> branchConstraints;
        int depth = 0; // 新增：追踪搜索树的深度
    };
    long long floorRational(const Rational& r)const{
        long long n=r.numerator();
        long long d=r.denominator();
        if(n>=0){
            return n/d;
        }else{
            return (n%d==0)?(n/d):(n/d-1);
        }
    }
    long long ceilRational(const Rational& r)const{
        long long n=r.numerator();
        long long d=r.denominator();
        if(n>=0){
            return (n%d==0)?(n/d):(n/d+1);
        }else{
            return n/d;
        }
    }
public:
    ILPBranchAndBound():sense(MAX),hasFeasibleIntegerSolution(false){}
    void setObjective(ObjSense s,const std::map<std::string,Rational>& terms){
        sense=s;
        objTerms=terms;
    }
    void addConstraint(const std::map<std::string,Rational>& terms,const std::string& op,const Rational& rhs){
        originalConstraints.push_back({terms,op,rhs});
    }
    void addIntegerVariable(const std::string& varName){
        integerVariables.insert(varName);
    }
    Simplex::Status solve(bool debug = false){ // 新增：调试开关
        hasFeasibleIntegerSolution=false;
        std::stack<SearchNode> stack;
        stack.push(SearchNode());
        int nodeCount = 0;
        while(!stack.empty()){
            SearchNode currNode=stack.top();
            stack.pop();
            nodeCount++;
            
            if(debug){
                std::cout<<"  [ILP Debug] 弹出 Node "<<nodeCount<<" | 树深度 "<<currNode.depth<<" | 附加约束数: "<<currNode.branchConstraints.size()<<"\n";
            }
            
            LPModel lp;
            lp.setObjective((LPModel::ObjSense)sense,objTerms);
            for(const auto& c:originalConstraints){
                lp.addConstraint(c.terms,c.op,c.rhs);
            }
            for(const auto& c:currNode.branchConstraints){
                lp.addConstraint(c.terms,c.op,c.rhs);
            }
            Simplex::Status lpStatus=lp.solve();
            if(lpStatus!=Simplex::OPTIMAL){
                if(debug) std::cout<<"    -> LP 无解或无界。已剪枝 (Pruned)。\n";
                continue;
            }
            Rational currentZ=lp.getObjectiveValue();
            if(debug) std::cout<<"    -> LP 松弛最优 Z = "<<currentZ<<"\n";
            
            if(hasFeasibleIntegerSolution){
                if(sense==MAX&&currentZ<=bestObjective){
                    if(debug) std::cout<<"    -> 价值剪枝。Z <= 已知最优整数解 ("<<bestObjective<<")\n";
                    continue;
                }
                if(sense==MIN&&currentZ>=bestObjective){
                    if(debug) std::cout<<"    -> 价值剪枝。Z >= 已知最优整数解 ("<<bestObjective<<")\n";
                    continue;
                }
            }
            auto lpSol=lp.getSolution();
            std::string branchVar="";
            Rational branchVal;
            for(const std::string& var:integerVariables){
                if(lpSol.count(var)){
                    if(lpSol[var].denominator()!=1){
                        branchVar=var;
                        branchVal=lpSol[var];
                        break;
                    }
                }
            }
            if(branchVar.empty()){
                bestObjective=currentZ;
                bestSolution=lpSol;
                hasFeasibleIntegerSolution=true;
                if(debug) std::cout<<"    -> 发现纯整数可行解！最优 Z 更新为: "<<bestObjective<<"\n";
            }else{
                if(debug) std::cout<<"    -> 触发分支 (Branching) 于 "<<branchVar<<" = "<<branchVal<<"\n";
                long long f=floorRational(branchVal);
                long long c=ceilRational(branchVal);
                SearchNode leftNode=currNode;
                leftNode.depth=currNode.depth+1; // 深度加深
                leftNode.branchConstraints.push_back({{{branchVar,Rational(1)}},"<=",Rational(f)});
                SearchNode rightNode=currNode;
                rightNode.depth=currNode.depth+1; // 深度加深
                rightNode.branchConstraints.push_back({{{branchVar,Rational(1)}},">=",Rational(c)});
                stack.push(leftNode);
                stack.push(rightNode);
            }
        }
        if(debug) std::cout<<"  [ILP Debug] 搜索完成。总计探索节点: "<<nodeCount<<"\n";
        return hasFeasibleIntegerSolution?Simplex::OPTIMAL:Simplex::INFEASIBLE;
    }
    Rational getObjectiveValue()const{
        if(!hasFeasibleIntegerSolution){
            throw std::logic_error("ILPBranchAndBound:No feasible integer solution.");
        }
        return bestObjective;
    }
    std::map<std::string,Rational> getSolution()const{
        if(!hasFeasibleIntegerSolution){
            throw std::logic_error("ILPBranchAndBound:No feasible integer solution.");
        }
        return bestSolution;
    }
};
#endif // ILP_BRANCH_AND_BOUND_H