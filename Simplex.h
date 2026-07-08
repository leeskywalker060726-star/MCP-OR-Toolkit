#ifndef SIMPLEX_H
#define SIMPLEX_H
#include<vector>
#include"Rational.h"
class Simplex{
public:
    enum Status{OPTIMAL,UNBOUNDED,INFEASIBLE};
    int m;
    int cols;
    std::vector<std::vector<Rational>> mat;
    std::vector<int> basis;
    Simplex(const std::vector<std::vector<Rational>>& initial_mat,const std::vector<int>& initial_basis):mat(initial_mat),basis(initial_basis){
        m=basis.size();
        cols=mat[0].size();
    }
    Status solve(){
        while(true){
            int inCol=getEnterCol();
            if(inCol==-1){
                return OPTIMAL;
            }
            int outRow=getLeaveRow(inCol);
            if(outRow==-1){
                return UNBOUNDED;
            }
            pivot(outRow,inCol);
        }
    }
private:
    int getEnterCol()const{
        int bestCol=-1;
        Rational minVal(0);
        for(int j=0;j<cols-1;++j){
            if(mat[m][j]<minVal){
                minVal=mat[m][j];
                bestCol=j;
            }
        }
        return bestCol;
    }
    int getLeaveRow(int col)const{
        int bestRow=-1;
        Rational minRatio(-1);
        bool found=false;
        for(int i=0;i<m;++i){
            if(mat[i][col]>Rational(0)){
                Rational ratio=mat[i].back()/mat[i][col];
                if(!found||ratio<minRatio){
                    minRatio=ratio;
                    bestRow=i;
                    found=true;
                }
            }
        }
        return bestRow;
    }
    void pivot(int r,int c){
        basis[r]=c;
        Rational p=mat[r][c];
        for(int j=0;j<cols;++j){
            mat[r][j]/=p;
        }
        for(int i=0;i<=m;++i){
            if(i!=r){
                Rational factor=mat[i][c];
                if(factor!=Rational(0)){
                    for(int j=0;j<cols;++j){
                        mat[i][j]-=factor*mat[r][j];
                    }
                }
            }
        }
    }
};
#endif // SIMPLEX_H