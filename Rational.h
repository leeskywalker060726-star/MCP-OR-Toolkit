#ifndef RATIONAL_H
#define RATIONAL_H
#include<numeric>
#include<cmath>
#include<iostream>
#include<stdexcept>
class Rational{
public:
    Rational(long long n=0,long long d=1):num_(n),den_(d){
        if(den_==0){
            throw std::invalid_argument("Rational:denominator cannot be zero.");
        }
        normalize();
    }
    long long numerator()const{return num_;}
    long long denominator()const{return den_;}
    explicit operator double()const{return static_cast<double>(num_)/static_cast<double>(den_);}
    Rational operator+()const{return *this;}
    Rational operator-()const{return Rational(-num_,den_);}
    Rational& operator+=(const Rational& rhs){
        long long new_num=num_*rhs.den_+rhs.num_*den_;
        long long new_den=den_*rhs.den_;
        num_=new_num;
        den_=new_den;
        normalize();
        return *this;
    }
    Rational& operator-=(const Rational& rhs){
        long long new_num=num_*rhs.den_-rhs.num_*den_;
        long long new_den=den_*rhs.den_;
        num_=new_num;
        den_=new_den;
        normalize();
        return *this;
    }
    Rational& operator*=(const Rational& rhs){
        num_*=rhs.num_;
        den_*=rhs.den_;
        normalize();
        return *this;
    }
    Rational& operator/=(const Rational& rhs){
        if(rhs.num_==0){
            throw std::domain_error("Rational:division by zero.");
        }
        num_*=rhs.den_;
        den_*=rhs.num_;
        normalize();
        return *this;
    }
    Rational& operator++(){*this+=1;return *this;}
    Rational operator++(int){Rational tmp(*this);*this+=1;return tmp;}
    Rational& operator--(){*this-=1;return *this;}
    Rational operator--(int){Rational tmp(*this);*this-=1;return tmp;}
    friend Rational operator+(Rational lhs,const Rational& rhs){lhs+=rhs;return lhs;}
    friend Rational operator-(Rational lhs,const Rational& rhs){lhs-=rhs;return lhs;}
    friend Rational operator*(Rational lhs,const Rational& rhs){lhs*=rhs;return lhs;}
    friend Rational operator/(Rational lhs,const Rational& rhs){lhs/=rhs;return lhs;}
    friend bool operator==(const Rational& lhs,const Rational& rhs){return lhs.num_==rhs.num_&&lhs.den_==rhs.den_;}
    friend bool operator!=(const Rational& lhs,const Rational& rhs){return !(lhs==rhs);}
    friend bool operator<(const Rational& lhs,const Rational& rhs){return lhs.num_*rhs.den_<rhs.num_*lhs.den_;}
    friend bool operator>(const Rational& lhs,const Rational& rhs){return rhs<lhs;}
    friend bool operator<=(const Rational& lhs,const Rational& rhs){return !(rhs<lhs);}
    friend bool operator>=(const Rational& lhs,const Rational& rhs){return !(lhs<rhs);}
    friend Rational abs(const Rational& r){return Rational(std::abs(r.num_),r.den_);}
    friend std::ostream& operator<<(std::ostream& os,const Rational& r){
        if(r.den_==1){
            os<<r.num_;
        }else{
            os<<r.num_<<"/"<<r.den_;
        }
        return os;
    }
private:
    void normalize(){
        if(den_<0){
            num_=-num_;
            den_=-den_;
        }
        auto g=std::gcd(std::abs(num_),den_);
        num_/=g;
        den_/=g;
        if(num_==0){
            den_=1;
        }
    }
    long long num_;
    long long den_;
};
#endif // RATIONAL_H