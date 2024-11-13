#include <bits/stdc++.h>
#include "../algo/debug.cpp"
#include "../FileSystem/DateMap.cpp" 
#include "../FileSystem/FindAbs.cpp" 
using namespace std;

int main(){
    // freopen("PaperInfo.txt", "w", stdout);
    string paper="0001001";
    
    dateMap d;
    FindAbs f(paper,d.m[paper]);
    cout<<d.m[paper]<<endl;;

    for(auto x:f.m){
        cout<<x.first<<" ---> "<<x.second<<endl<<endl;;
    }

    return 0;
}