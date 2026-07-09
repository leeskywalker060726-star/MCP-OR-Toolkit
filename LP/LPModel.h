#ifndef LPMODEL_H
#define LPMODEL_H
#include<string>
#include<vector>
#include<map>
#include<stdexcept>
#include<algorithm>
#include"Rational.h"
#include"Simplex.h"
class LPModel{
public:
    enum ObjSense{MAX,MIN};
private:
    ObjSense sense;
    std::map<std::string,int> varToReg;
    std::vector<std::string> regToVar;
    std::vector<int> artificialRegs;
    std::map<int,Rational> objIR;
    struct ConstraintIR{
        std::map<int,Rational> terms;
        std::string op;
        Rational rhs;
    };
    std::vector<ConstraintIR> constraintsIR;
    std::vector<std::vector<Rational>> finalTableau;
    Rational optimalValue;
    std::vector<Rational> optimalSolution;
    
    // 补丁 1：定义一个计数器，给底层自动生成的松弛变量起名字
    int hiddenVarCounter = 0; 
    
    int allocateReg(const std::string& name=""){
        int regId=regToVar.size();
        regToVar.push_back(name);
        if(!name.empty()){
            varToReg[name]=regId;
        }
        return regId;
    }
    int getOrAllocateReg(const std::string& name){
        if(varToReg.find(name)==varToReg.end()){
            return allocateReg(name);
        }
        return varToReg[name];
    }
public:
    LPModel():sense(MAX){}
    void setObjective(ObjSense s,const std::map<std::string,Rational>& terms){
        sense=s;
        objIR.clear();
        for(const auto& pair:terms){
            objIR[getOrAllocateReg(pair.first)]=pair.second;
        }
    }
    void addConstraint(std::map<std::string,Rational> terms,std::string op,Rational rhs){
        if(rhs<Rational(0)){
            for(auto& pair:terms){
                pair.second=-pair.second;
            }
            rhs=-rhs;
            if(op=="<="){
                op=">=";
            }else if(op==">="){
                op="<=";
            }
        }
        if(op!="<="&&op!=">="&&op!="="){
            throw std::invalid_argument("LPModel:Unsupported operator.");
        }
        ConstraintIR ir;
        ir.op=op;
        ir.rhs=rhs;
        for(const auto& pair:terms){
            ir.terms[getOrAllocateReg(pair.first)]=pair.second;
        }
        constraintsIR.push_back(ir);
    }
    Simplex::Status solve(){
        int m=constraintsIR.size();
        if(m==0||varToReg.empty()){
            throw std::logic_error("LPModel:Model is empty.");
        }
        std::vector<int> basis(m);
        artificialRegs.clear();
        for(int i=0;i<m;++i){
            auto& cons=constraintsIR[i];
            if(cons.op=="<="){
                // 补丁 2：给原本没名字的松弛变量强行赋名，让它可以在下一轮当做普通变量被约束
                int sReg=allocateReg("_S" + std::to_string(++hiddenVarCounter));
                cons.terms[sReg]=Rational(1);
                basis[i]=sReg;
            }else if(cons.op==">="){
                int eReg=allocateReg("_E" + std::to_string(++hiddenVarCounter));
                int aReg=allocateReg("_A" + std::to_string(++hiddenVarCounter));
                cons.terms[eReg]=Rational(-1);
                cons.terms[aReg]=Rational(1);
                basis[i]=aReg;
                artificialRegs.push_back(aReg);
            }else if(cons.op=="="){
                int aReg=allocateReg();
                cons.terms[aReg]=Rational(1);
                basis[i]=aReg;
                artificialRegs.push_back(aReg);
            }
        }
        int totalRegs=regToVar.size();
        std::vector<std::vector<Rational>> mat(m+1,std::vector<Rational>(totalRegs+1,Rational(0)));
        for(int i=0;i<m;++i){
            for(const auto& pair:constraintsIR[i].terms){
                mat[i][pair.first]=pair.second;
            }
            mat[i].back()=constraintsIR[i].rhs;
        }
        if(!artificialRegs.empty()){
            for(int aReg:artificialRegs){
                mat[m][aReg]=Rational(1);
            }
            for(int i=0;i<m;++i){
                if(std::find(artificialRegs.begin(),artificialRegs.end(),basis[i])!=artificialRegs.end()){
                    for(int j=0;j<=totalRegs;++j){
                        mat[m][j]-=mat[i][j];
                    }
                }
            }
            Simplex phase1(mat,basis);
            phase1.solve();
            if(phase1.mat[m].back()<Rational(0)){
                return Simplex::INFEASIBLE;
            }
            mat=phase1.mat;
            basis=phase1.basis;
            for(int j=0;j<=totalRegs;++j){
                mat[m][j]=Rational(0);
            }
            for(int i=0;i<m;++i){
                if(std::find(artificialRegs.begin(),artificialRegs.end(),basis[i])!=artificialRegs.end()){
                    int pivotCol=-1;
                    for(int j=0;j<totalRegs;++j){
                        if(std::find(artificialRegs.begin(),artificialRegs.end(),j)==artificialRegs.end()&&mat[i][j]!=Rational(0)){
                            pivotCol=j;
                            break;
                        }
                    }
                    if(pivotCol!=-1){
                        Rational p=mat[i][pivotCol];
                        for(int j=0;j<=totalRegs;++j){
                            mat[i][j]/=p;
                        }
                        for(int r=0;r<=m;++r){
                            if(r!=i){
                                Rational factor=mat[r][pivotCol];
                                if(factor!=Rational(0)){
                                    for(int j=0;j<=totalRegs;++j){
                                        mat[r][j]-=factor*mat[i][j];
                                    }
                                }
                            }
                        }
                        basis[i]=pivotCol;
                    }
                }
            }
            for(int aReg:artificialRegs){
                for(int i=0;i<=m;++i){
                    mat[i][aReg]=Rational(0);
                }
            }
        }
        for(const auto& pair:objIR){
            mat[m][pair.first]=(sense==MAX)?-pair.second:pair.second;
        }
        for(int i=0;i<m;++i){
            int currentBasisReg=basis[i];
            Rational factor=mat[m][currentBasisReg];
            if(factor!=Rational(0)){
                for(int j=0;j<=totalRegs;++j){
                    mat[m][j]-=factor*mat[i][j];
                }
            }
        }
        Simplex phase2(mat,basis);
        Simplex::Status finalStatus=phase2.solve();
        if(finalStatus==Simplex::OPTIMAL){
            finalTableau=phase2.mat;
            optimalValue=phase2.mat[m].back();
            if(sense==MIN){
                optimalValue=-optimalValue;
            }
            optimalSolution.assign(totalRegs,Rational(0));
            for(int i=0;i<m;++i){
                optimalSolution[phase2.basis[i]]=phase2.mat[i].back();
            }
        }
        return finalStatus;
    }
    Rational getObjectiveValue()const{return optimalValue;}
    std::map<std::string,Rational> getSolution()const{
        std::map<std::string,Rational> result;
        for(const auto& pair:varToReg){
            result[pair.first]=optimalSolution[pair.second];
        }
        return result;
    }
    const std::vector<std::string>& getRegToVar() const { return regToVar; }
    const std::vector<std::vector<Rational>>& getFinalTableau() const { return finalTableau; }
};
#endif // LPMODEL_H