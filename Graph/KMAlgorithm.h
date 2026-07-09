#ifndef KM_ALGORITHM_H
#define KM_ALGORITHM_H
#include<string>
#include<vector>
#include<map>
#include<iostream>
#include<algorithm>
#include<limits>
class KMAlgorithm{
private:
    std::map<std::string,int> leftNodes;
    std::vector<std::string> leftNames;
    std::map<std::string,int> rightNodes;
    std::vector<std::string> rightNames;
    std::vector<std::vector<long long>> weights;
    std::vector<long long> lx;
    std::vector<long long> ly;
    std::vector<int> matchRight;
    std::vector<bool> visLeft;
    std::vector<bool> visRight;
    std::vector<long long> slack;
    int getOrAddLeft(const std::string& name){
        if(leftNodes.find(name)==leftNodes.end()){
            leftNodes[name]=leftNames.size();
            leftNames.push_back(name);
        }
        return leftNodes[name];
    }
    int getOrAddRight(const std::string& name){
        if(rightNodes.find(name)==rightNodes.end()){
            rightNodes[name]=rightNames.size();
            rightNames.push_back(name);
        }
        return rightNodes[name];
    }
    bool dfs(int u){
        visLeft[u]=true;
        int n=rightNames.size();
        for(int v=0;v<n;++v){
            if(visRight[v])continue;
            long long gap=lx[u]+ly[v]-weights[u][v];
            if(gap==0){
                visRight[v]=true;
                if(matchRight[v]==-1||dfs(matchRight[v])){
                    matchRight[v]=u;
                    return true;
                }
            }else{
                slack[v]=std::min(slack[v],gap);
            }
        }
        return false;
    }
public:
    KMAlgorithm(){}
    void addEdge(const std::string& leftName,const std::string& rightName,long long weight){
        int u=getOrAddLeft(leftName);
        int v=getOrAddRight(rightName);
        int maxSize=std::max(leftNames.size(),rightNames.size());
        weights.resize(maxSize,std::vector<long long>(maxSize,0));
        for(auto& row:weights)row.resize(maxSize,0);
        weights[u][v]=weight;
    }
    long long solve(bool debug=false){
        int n=std::max(leftNames.size(),rightNames.size());
        if(n==0)return 0;
        lx.assign(n,0);
        ly.assign(n,0);
        matchRight.assign(n,-1);
        for(int u=0;u<n;++u){
            for(int v=0;v<n;++v){
                lx[u]=std::max(lx[u],weights[u][v]);
            }
        }
        for(int i=0;i<n;++i){
            if(debug)std::cout<<"\n[KM Debug] 正在为 ["<<leftNames[i]<<"] 寻找匹配...\n";
            int loopCount=0;
            while(true){
                loopCount++;
                visLeft.assign(n,false);
                visRight.assign(n,false);
                slack.assign(n,std::numeric_limits<long long>::max()); // 修复：必须在每次重新 DFS 前重置 Slack
                if(dfs(i)){
                    if(debug)std::cout<<"  -> 成功找到增广路！(本轮共尝试 "<<loopCount<<" 次)\n";
                    break;
                }
                long long d=std::numeric_limits<long long>::max();
                for(int v=0;v<n;++v){
                    if(!visRight[v]){
                        d=std::min(d,slack[v]);
                    }
                }
                if(d==std::numeric_limits<long long>::max() || d==0) break; // 防死循环兜底
                if(debug)std::cout<<"  -> 匹配卡壳，触发调价，最小落差 d = "<<d<<"\n";
                for(int u=0;u<n;++u){
                    if(visLeft[u])lx[u]-=d;
                }
                for(int v=0;v<n;++v){
                    if(visRight[v])ly[v]+=d;
                }
            }
        }
        long long maxWeight=0;
        for(int v=0;v<n;++v){
            if(matchRight[v]!=-1){
                maxWeight+=weights[matchRight[v]][v];
            }
        }
        return maxWeight;
    }
    std::map<std::string,std::string> getMatches()const{
        std::map<std::string,std::string> results;
        for(int v=0;v<matchRight.size();++v){
            int u=matchRight[v];
            if(u!=-1&&u<leftNames.size()&&v<rightNames.size()){
                results[leftNames[u]]=rightNames[v];
            }
        }
        return results;
    }
};
#endif // KM_ALGORITHM_H