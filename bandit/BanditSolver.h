#ifndef BANDIT_SOLVER_H
#define BANDIT_SOLVER_H
#include<string>
#include<map>
#include<vector>
#include<cmath>
#include<random>
#include<stdexcept>
#include<algorithm>
#include<limits>
struct ArmState{
    int pulls;
    double total_reward;
};
struct BanditResult{
    std::string status;
    std::string chosen_arm;
    std::map<std::string,double> scores;
    std::string message;
};
class BanditSolver{
private:
    std::string strategy;
    std::map<std::string,double> hyperparams;
    std::map<std::string,ArmState> arms;
    std::mt19937 rng;
    double getHyperparam(const std::string& key,double default_val){
        if(hyperparams.find(key)!=hyperparams.end()){
            return hyperparams[key];
        }
        return default_val;
    }
    BanditResult solveEpsilonGreedy(){
        BanditResult result;
        double epsilon=getHyperparam("epsilon",0.1);
        std::uniform_real_distribution<double> dist(0.0,1.0);
        bool explore=(dist(rng)<epsilon);
        std::string best_arm="";
        double best_mean=-std::numeric_limits<double>::infinity();
        std::vector<std::string> all_arm_names;
        for(const auto& pair:arms){
            all_arm_names.push_back(pair.first);
            double mean_reward=(pair.second.pulls>0)?(pair.second.total_reward/pair.second.pulls):0.0;
            result.scores[pair.first]=mean_reward;
            if(mean_reward>best_mean){
                best_mean=mean_reward;
                best_arm=pair.first;
            }
        }
        if(explore){
            std::uniform_int_distribution<int> int_dist(0,all_arm_names.size()-1);
            result.chosen_arm=all_arm_names[int_dist(rng)];
            result.message="Exploration triggered (random choice).";
        }else{
            result.chosen_arm=best_arm;
            result.message="Exploitation triggered (chose arm with max mean reward).";
        }
        result.status="SUCCESS";
        return result;
    }
    BanditResult solveUCB(){
        BanditResult result;
        double c=getHyperparam("c",1.414);
        int total_pulls=0;
        for(const auto& pair:arms){
            total_pulls+=pair.second.pulls;
        }
        std::string best_arm="";
        double max_ucb=-std::numeric_limits<double>::infinity();
        for(const auto& pair:arms){
            const std::string& name=pair.first;
            const ArmState& state=pair.second;
            if(state.pulls==0){
                result.scores[name]=std::numeric_limits<double>::max();
                if(result.scores[name]>max_ucb){
                    max_ucb=result.scores[name];
                    best_arm=name;
                }
                continue;
            }
            double mean_reward=state.total_reward/state.pulls;
            double ucb_value=mean_reward+c*std::sqrt(std::log(total_pulls)/state.pulls);
            result.scores[name]=ucb_value;
            if(ucb_value>max_ucb){
                max_ucb=ucb_value;
                best_arm=name;
            }
        }
        result.chosen_arm=best_arm;
        result.status="SUCCESS";
        result.message="UCB calculated. Chose arm with max upper confidence bound.";
        return result;
    }
    BanditResult solveThompsonSampling(){
        BanditResult result;
        std::string best_arm="";
        double max_sample=-1.0;
        for(const auto& pair:arms){
            const std::string& name=pair.first;
            const ArmState& state=pair.second;
            double alpha=1.0+state.total_reward;
            double beta=1.0+(state.pulls-state.total_reward);
            std::gamma_distribution<double> gamma_alpha(alpha,1.0);
            std::gamma_distribution<double> gamma_beta(beta,1.0);
            double a_sample=gamma_alpha(rng);
            double b_sample=gamma_beta(rng);
            double beta_sample=a_sample/(a_sample+b_sample);
            result.scores[name]=beta_sample;
            if(beta_sample>max_sample){
                max_sample=beta_sample;
                best_arm=name;
            }
        }
        result.chosen_arm=best_arm;
        result.status="SUCCESS";
        result.message="Thompson Sampling applied using Beta distribution.";
        return result;
    }
public:
    BanditSolver(){}
    BanditResult solve(const std::string& strat,const std::map<std::string,ArmState>& input_arms,const std::map<std::string,double>& hparams){
        strategy=strat;
        arms=input_arms;
        hyperparams=hparams;
        int seed=static_cast<int>(getHyperparam("seed",42.0));
        rng.seed(seed);
        if(arms.empty()){
            return {"ERROR","",{},"No arms provided."};
        }
        if(strategy=="EPSILON_GREEDY"){
            return solveEpsilonGreedy();
        }else if(strategy=="UCB"){
            return solveUCB();
        }else if(strategy=="THOMPSON_SAMPLING"){
            return solveThompsonSampling();
        }else{
            return {"ERROR","",{},"Unsupported Bandit strategy: "+strategy};
        }
    }
};
#endif