#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

// ============================================================================
// 匈牙利算法 (Hungarian Algorithm) - 无权二分图最大匹配
// 解决问题：如何在一个二分图中，找到连线数最多的“一对一”完美匹配
// ============================================================================
class Hungarian {
private:
    std::map<std::string, int> leftNodes;
    std::vector<std::string> leftNames;
    
    std::map<std::string, int> rightNodes;
    std::vector<std::string> rightNames;
    
    // 邻接表：adj[u] 存储了左侧节点 u (如张三) 能够接受的所有右侧节点 v (如单1) 的 ID
    std::vector<std::vector<int>> adj;
    
    // 核心状态数组 1：matchRight[v] = u
    // 记录右侧节点 v 当前的匹配对象是左侧节点 u。如果 v 是单身，则值为 -1。
    std::vector<int> matchRight;
    
    // 核心状态数组 2：vis[v] = true
    // 记录在【当前这一轮】为某个左侧节点找对象时，右侧节点 v 是否已经被询问过（防止无限套娃死循环）
    std::vector<bool> vis;

    // 辅助函数：将字符串名字转化为内部整数 ID
    int getOrAddLeft(const std::string& name) {
        if (leftNodes.find(name) == leftNodes.end()) {
            leftNodes[name] = leftNames.size();
            leftNames.push_back(name);
        }
        return leftNodes[name];
    }

    int getOrAddRight(const std::string& name) {
        if (rightNodes.find(name) == rightNodes.end()) {
            rightNodes[name] = rightNames.size();
            rightNames.push_back(name);
        }
        return rightNodes[name];
    }

    // ========================================================================
    // 🔥 TODO 1: 核心“协商让位”逻辑 (增广路搜索) 🔥
    // ========================================================================
    // 尝试为左侧节点 u 寻找一个匹配对象。如果成功找到并绑定，返回 true。
    bool dfs(int u) {
        // 提示：
        // 1. 遍历 u 能接受的所有右侧节点 v (都在 adj[u] 里面)。
        // 2. 如果 v 在这一轮还没有被访问过 (!vis[v])：
        //      a. 立刻标记 v 为已访问 (vis[v] = true)。
        //      b. 检查两种可以成功牵手的情况：
        //         情况一：v 目前单身 (matchRight[v] == -1)
        //         情况二：v 有现任，但现任能够通过递归调用 dfs() 找到别的备胎。
        //      c. 如果满足上述任一情况，请将 v 的现任更新为 u，并返回 true。
        // 3. 如果把所有备胎都问了一遍还是不行，返回 false。
        
        // ---> 请在此处实现你的 DFS 逻辑 <---
        for(auto v : adj[u]) {
            if(!vis[v]) {
                vis[v] = true; // 标记 v 已访问
                // 情况一：v 单身，或者情况二：v 的现任可以找到别的备胎
                if(matchRight[v] == -1 || dfs(matchRight[v])) {
                    matchRight[v] = u; // 绑定 v 的现任为 u
                    return true; // 成功找到匹配
                }
            }
        }
        return false; // 占位符
    }

public:
    Hungarian() {}

    // 添加一条从左侧节点到右侧节点的连线 (表示可以匹配)
    void addEdge(const std::string& leftName, const std::string& rightName) {
        int u = getOrAddLeft(leftName);
        int v = getOrAddRight(rightName);
        if (u >= adj.size()) {
            adj.resize(u + 1);
        }
        adj[u].push_back(v);
    }

    // ========================================================================
    // 🔥 TODO 2: 算法主循环 🔥
    // ========================================================================
    // 执行算法，返回最大匹配的对数
    int solve() {
        int numLeft = leftNames.size();
        int numRight = rightNames.size();
        
        // 初始化：所有人均为单身 (-1)
        matchRight.assign(numRight, -1); 
        
        int maxMatches = 0;

        // 提示：
        // 1. 写一个 for 循环，遍历所有的左侧节点 i (从 0 到 numLeft - 1)。
        // 2. 在每次调用 dfs(i) 之前，必须把 vis 数组清空（全部设为 false），
        //    确保在为当前节点找增广路时，右侧节点都能被正常访问。
        //    (可以使用 std::fill 或 vis.assign(numRight, false))
        // 3. 调用 dfs(i)，如果返回 true，说明匹配对数多了一对，maxMatches 加 1。
        
        // ---> 请在此处实现你的主循环逻辑 <---
        for (int i = 0; i < numLeft; ++i) {
            vis.assign(numRight, false); // 清空访问标记
            if (dfs(i)) {
                maxMatches++; // 成功匹配，计数加 1
            }
        }
        
        
        return maxMatches;
    }

    // ========================================================================
    // 以下方法无需修改，用于对外输出结果
    // ========================================================================
    std::map<std::string, std::string> getMatches() const {
        std::map<std::string, std::string> results;
        for (int v = 0; v < matchRight.size(); ++v) {
            int u = matchRight[v];
            if (u != -1) {
                // 返回映射：左侧名字 -> 右侧名字
                results[leftNames[u]] = rightNames[v];
            }
        }
        return results;
    }
};

#endif // HUNGARIAN_H