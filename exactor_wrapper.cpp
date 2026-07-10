#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<string>
#include<stdexcept>
#include<cmath>
#include"LP/Rational.h"
#include"LP/LPModel.h"
#include"ILP/ILPBranchAndBound.h"
#include"ILP/ILPGomoryCut.h"
#include"Graph/Hungarian.h"
#include"Graph/KMAlgorithm.h"

namespace py=pybind11;

// 辅助函数：将 Python 的 int 或 float 转化为 C++ 的 Rational 有理数
Rational parseRational(const py::handle& obj){
    if(py::isinstance<py::int_>(obj)){
        return Rational(obj.cast<long long>());
    }else{
        double val=obj.cast<double>();
        long long den=10000; 
        long long num=std::round(val*den);
        return Rational(num,den);
    }
}

// ============================================================================
// ExactOR 万能分发器 (Dispatcher)
// ============================================================================
py::dict solve_dispatcher(const std::string& algo_type, py::dict payload){
    py::dict result;
    try{
        if(algo_type=="LP" || algo_type=="ILP_BB" || algo_type=="GOMORY"){
            // 提取目标函数
            std::map<std::string,Rational> objTerms;
            py::dict pyObj=payload["objective"].cast<py::dict>();
            for(auto item:pyObj) objTerms[item.first.cast<std::string>()]=parseRational(item.second);
            int sense=(payload.contains("sense")&&payload["sense"].cast<std::string>()=="MIN")?1:0; 
            
            // 提取约束条件
            struct ConstraintData{std::map<std::string,Rational> terms;std::string op;Rational rhs;};
            std::vector<ConstraintData> constraints;
            py::list pyCons=payload["constraints"].cast<py::list>();
            for(auto item:pyCons){
                py::dict c=item.cast<py::dict>();
                std::map<std::string,Rational> terms;
                py::dict cTerms=c["terms"].cast<py::dict>();
                for(auto t:cTerms) terms[t.first.cast<std::string>()]=parseRational(t.second);
                constraints.push_back({terms,c["op"].cast<std::string>(),parseRational(c["rhs"])});
            }
            
            // 执行具体算法分发
            Simplex::Status status;
            Rational optZ;
            std::map<std::string,Rational> sol;
            
            if(algo_type=="LP"){
                LPModel model;
                model.setObjective((LPModel::ObjSense)sense,objTerms);
                for(const auto& c:constraints) model.addConstraint(c.terms,c.op,c.rhs);
                status=model.solve();
                if(status==Simplex::OPTIMAL){optZ=model.getObjectiveValue();sol=model.getSolution();}
            }else if(algo_type=="ILP_BB"){
                ILPBranchAndBound model;
                model.setObjective((ILPBranchAndBound::ObjSense)sense,objTerms);
                for(const auto& c:constraints) model.addConstraint(c.terms,c.op,c.rhs);
                if(payload.contains("integer_variables")){
                    py::list intVars=payload["integer_variables"].cast<py::list>();
                    for(auto v:intVars) model.addIntegerVariable(v.cast<std::string>());
                }
                status=model.solve();
                if(status==Simplex::OPTIMAL){optZ=model.getObjectiveValue();sol=model.getSolution();}
            }else if(algo_type=="GOMORY"){
                ILPGomoryCut model;
                model.setObjective((ILPGomoryCut::ObjSense)sense,objTerms);
                for(const auto& c:constraints) model.addConstraint(c.terms,c.op,c.rhs);
                status=model.solve(false);
                if(status==Simplex::OPTIMAL){optZ=model.getObjectiveValue();sol=model.getSolution();}
            }
            
            // 封装结果
            if(status==Simplex::OPTIMAL){
                result["status"]="OPTIMAL";
                result["objective_value"]=double(optZ);
                py::dict pySol;
                for(const auto& pair:sol) pySol[pair.first.c_str()]=double(pair.second);
                result["solution"]=pySol;
            }else if(status==Simplex::INFEASIBLE){result["status"]="INFEASIBLE";}
            else{result["status"]="UNBOUNDED";}
        }
        else if(algo_type=="HUNGARIAN"){
            Hungarian hungarian;
            py::list pyEdges=payload["edges"].cast<py::list>();
            for(auto item:pyEdges){
                py::dict edge=item.cast<py::dict>();
                hungarian.addEdge(edge["u"].cast<std::string>(), edge["v"].cast<std::string>());
            }
            int maxMatches = hungarian.solve();
            result["status"]="OPTIMAL";
            result["max_matches"]=maxMatches;
            py::dict matches;
            for(const auto& pair:hungarian.getMatches()) matches[pair.first.c_str()]=pair.second;
            result["matches"]=matches;
        }
        else if(algo_type=="KM"){
            KMAlgorithm km;
            py::list pyEdges=payload["edges"].cast<py::list>();
            for(auto item:pyEdges){
                py::dict edge=item.cast<py::dict>();
                km.addEdge(edge["u"].cast<std::string>(),edge["v"].cast<std::string>(),edge["w"].cast<long long>());
            }
            long long maxWeight=km.solve(false);
            result["status"]="OPTIMAL";
            result["max_weight"]=maxWeight;
            py::dict matches;
            for(const auto& pair:km.getMatches()) matches[pair.first.c_str()]=pair.second;
            result["matches"]=matches;
        }
        else{
            throw std::invalid_argument("ExactOR: Unsupported algorithm type: "+algo_type);
        }
    }catch(const std::exception& e){
        result["status"]="ERROR";
        result["message"]=e.what();
    }
    return result;
}

// Pybind11 模块定义
PYBIND11_MODULE(exactor,m){
    m.doc()="ExactOR: A High-Precision Operations Research Engine via Pybind11";
    m.def("solve",&solve_dispatcher,"Universal dispatcher for operations research algorithms",py::arg("algo_type"),py::arg("payload"));
}