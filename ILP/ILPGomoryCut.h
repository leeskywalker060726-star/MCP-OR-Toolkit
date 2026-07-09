#ifndef ILP_GOMORY_CUT_H
#define ILP_GOMORY_CUT_H
#include<string>
#include<vector>
#include<map>
#include<iostream>
#include"../LP/Rational.h"
#include"../LP/LPModel.h"
class ILPGomoryCut{
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
    std::vector<ConstraintDef> constraints;
    
    Rational optimalValue;
    std::map<std::string,Rational> optimalSolution;
    
    Rational getFractionalPart(const Rational& r)const{
        long long n=r.numerator();
        long long d=r.denominator();
        long long floor_val;
        if(n>=0){
            floor_val=n/d;
        }else{
            floor_val=(n%d==0)?(n/d):(n/d-1);
        }
        return r-Rational(floor_val,1);
    }
public:
    ILPGomoryCut():sense(MAX){}
    void setObjective(ObjSense s,const std::map<std::string,Rational>& terms){
        sense=s;
        objTerms=terms;
    }
    void addConstraint(const std::map<std::string,Rational>& terms,const std::string& op,const Rational& rhs){
        constraints.push_back({terms,op,rhs});
    }
    Simplex::Status solve(bool debug=false){
        int cutIteration=0;
        while(true){
            cutIteration++;
            if(debug)std::cout<<"\n[Gomory] 第 "<<cutIteration<<" 轮 LP 松弛求解...\n";
            LPModel lp;
            lp.setObjective((LPModel::ObjSense)sense,objTerms);
            for(const auto& c:constraints){
                lp.addConstraint(c.terms,c.op,c.rhs);
            }
            Simplex::Status status=lp.solve();
            if(status!=Simplex::OPTIMAL){
                if(debug)std::cout<<"[Gomory] 底层 LP 无解或无界，终止。\n";
                return status;
            }
            auto solution=lp.getSolution();
            if(debug)std::cout<<"  -> 当前 Z = "<<lp.getObjectiveValue()<<"\n";
            
            // 验证是否达到纯整数解（仅需检查用户添加的原生变量）
            bool allInteger=true;
            for(const auto& pair:solution){
                // 忽略底层暗中生成的变量
                if(pair.first.find("_S")==0 || pair.first.find("_E")==0 || pair.first.find("_A")==0){
                    continue;
                }
                if(pair.second.denominator()!=1){
                    allInteger=false;
                    break;
                }
            }
            
            if(allInteger){
                if(debug)std::cout<<"[Gomory] 找到全局最优整数解！共切了 "<<cutIteration-1<<" 刀。\n";
                optimalValue = lp.getObjectiveValue();
                optimalSolution = solution;
                return Simplex::OPTIMAL;
            }
            
            const auto& tableau=lp.getFinalTableau();
            const auto& regToVar=lp.getRegToVar();
            int cutRow=-1;
            int rows=tableau.size()-1;
            
            // 简单粗暴：直接寻找包含分数的行（去掉对 basis 的依赖）
            for(int i=0;i<rows;++i){
                if(tableau[i].back().denominator()!=1){
                    cutRow=i;
                    break;
                }
            }
            
            if(cutRow==-1){
                optimalValue = lp.getObjectiveValue();
                optimalSolution = solution;
                return Simplex::OPTIMAL;
            }
            
            // 构造原始带有松弛变量的割
            std::map<std::string,Rational> rawCutTerms;
            Rational rawCutRhs=-getFractionalPart(tableau[cutRow].back());
            for(int j=0;j<tableau[cutRow].size()-1;++j){
                Rational f_j=getFractionalPart(tableau[cutRow][j]);
                if(f_j!=Rational(0)){
                    rawCutTerms[regToVar[j]]=-f_j;
                }
            }
            
            // ========================================================
            // [核心修复]：代数消元器
            // 追踪当前 LPModel 的所有松弛变量定义，并将其反向替换回原变量
            // ========================================================
            std::map<std::string,std::pair<std::map<std::string,Rational>,Rational>> slackDefs;
            std::map<std::string,std::pair<std::map<std::string,Rational>,Rational>> surplusDefs;
            int hiddenCounter=0;
            for(const auto& orig_c:constraints){
                std::map<std::string,Rational> c_terms=orig_c.terms;
                std::string c_op=orig_c.op;
                Rational c_rhs=orig_c.rhs;
                
                // 模拟 LPModel 对负 RHS 的预处理
                if(c_rhs<Rational(0)){
                    for(auto& p:c_terms) p.second=-p.second;
                    c_rhs=-c_rhs;
                    if(c_op=="<=") c_op=">=";
                    else if(c_op==">=") c_op="<=";
                }
                
                if(c_op=="<="){
                    std::string sName="_S"+std::to_string(++hiddenCounter);
                    slackDefs[sName]={c_terms,c_rhs}; // _Sk = rhs - terms
                }else if(c_op==">="){
                    std::string eName="_E"+std::to_string(++hiddenCounter);
                    surplusDefs[eName]={c_terms,c_rhs}; // _Ek = terms - rhs
                    hiddenCounter++; // 模拟跳过人工变量
                }else if(c_op=="="){
                    hiddenCounter++; // 模拟跳过人工变量
                }
            }
            
            // 执行消元替换
            std::map<std::string,Rational> pureCutTerms;
            Rational pureCutRhs=rawCutRhs;
            for(const auto& pair:rawCutTerms){
                const std::string& vName=pair.first;
                Rational coeff=pair.second;
                if(slackDefs.count(vName)){
                    const auto& def=slackDefs[vName];
                    pureCutRhs-=coeff*def.second;
                    for(const auto& t:def.first) pureCutTerms[t.first]-=coeff*t.second;
                }else if(surplusDefs.count(vName)){
                    const auto& def=surplusDefs[vName];
                    pureCutRhs+=coeff*def.second;
                    for(const auto& t:def.first) pureCutTerms[t.first]+=coeff*t.second;
                }else if(vName.find("_A")==0){
                    continue; // 人工变量直接剔除
                }else{
                    pureCutTerms[vName]+=coeff;
                }
            }
            
            // 剔除系数抵消为 0 的项，保持稀疏
            std::map<std::string,Rational> finalCutTerms;
            for(const auto& pair:pureCutTerms){
                if(pair.second!=Rational(0)){
                    finalCutTerms[pair.first]=pair.second;
                }
            }
            
            if(debug)std::cout<<"  -> 成功构造无污染的纯决策变量割平面！\n";
            constraints.push_back({finalCutTerms,"<=",pureCutRhs});
        }
    }
    
    Rational getObjectiveValue()const{return optimalValue;}
    std::map<std::string,Rational> getSolution()const{return optimalSolution;}
};
#endif // ILP_GOMORY_CUT_H