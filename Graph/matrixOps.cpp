#include <bits/stdc++.h>
#include "../algo/debug.cpp"
using namespace std;

class matrixOperation
{
public:
   vvlld copy(vvll org)
   {
      vvlld temp(org.size(), vlld(org[0].size()));
      for (int i = 0; i < org.size(); i++){
         for (int j = 0; j < org[0].size(); j++){
            temp[i][j] = (lld)org[i][j];
         }
      }
      return temp;
   }

   vvlld scalarMul(lld d, vvlld R){
      for (int i = 0; i < R.size(); i++){
         for (int j = 0; j < R[0].size(); j++){
            R[i][j] *= d;
         }
      }
      return R; 
   }

   vvlld add(vvlld E, vvlld R){
      for (int i = 0; i < E.size(); i++){
         for (int j = 0; j < E[0].size(); j++){
            R[i][j] += E[i][j];
         }
      }
      return R;
   }

   vvlld mul(vvlld mt1, vvlld mt2){
      vvlld mat1 = mt1, mat2 = mt2;
      ll m1 = mat1.size(), m2 = mat2.size(), n1 = mat1[0].size(), n2 = mat2[0].size();

      vvlld ans(m1, vlld(n2));
      if(n1==m2){
         for (int i = 0; i < m1; i++){
            for (int j = 0; j < n2; j++){
               for (int k = 0; k < n1; k++){
                  ans[i][j] += mat1[i][k] * mat2[k][j];
               }
            }
         }
      }
      return ans;
   }
};
