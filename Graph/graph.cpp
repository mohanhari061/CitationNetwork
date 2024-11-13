#include <bits/stdc++.h>
#include "./matrixOps.cpp"
namespace fs = filesystem;

using namespace std;


class graph {
public:
    ll n, m;
    bool directed;
    map<ll,ll> rmp;
    vvll adjList;
    vvll adjMat, d;
    matrixOperation matOps;  

    graph() {}

    graph(ll n, vvll edges, map<ll,ll> rmp, bool directed = false) {
        this->n = n;
        this->m = edges.size();
        this->directed = directed;
        this->adjMat = vvll(n, vll(n, 0));
        this->adjList = vvll(n);
        this->d = vvll(n, vll(n, 1e9));
        this->rmp = rmp;

        for (int i = 0; i < m; i++) {
            adjList[edges[i][0]].push_back(edges[i][1]);
            adjMat[edges[i][0]][edges[i][1]] = 1;
            if (!directed) {
                adjList[edges[i][1]].push_back(edges[i][0]);
                adjMat[edges[i][1]][edges[i][0]] = 1;
            }
        }

        // cout << "constructor called";
    }
    void publication(ll paper){
        string filePath = "H:/MCA/3rd Sem/3rdSem/SNA/CitationNetwork/Dataset/Sample/1000nodes/docdata/" + to_string(paper) + ".abs";
        ifstream file(filePath);
        string content = "";
        if (file.is_open()){
            string line;
            while (getline(file, line))
            {
                content += line +"\n";
            }
            file.close();
            cout<<content<<endl;
        }
        else{
            cout << "Not Found"<<endl;;
        }
    }
    void display(){
        cout<<"Adjacency List"<<endl;
        for(int i=0;i<n;i++){
            // cout<<rmp[i]<<" :: ";
            cout<<i<<" :: ";
            for(auto ele:adjList[i]){
                // cout<<rmp[ele]<<" ";
                cout<<ele<<" ";
            }
            cout<<nline;
            
        }
        cout<<nline<<nline;
        cout<<"Adjacency Matrix"<<endl;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                cout<<this->adjMat[i][j]<<" ";
            }
            cout<<nline;
        }
    }
    
    void degreeDistribution(){
        map<ll,lld> mpd;
        vll in(n),out(n),deg(n);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(adjMat[i][j]){
                out[i]++;
                in[j]++;
                }
            }
        }
        for(int i=0;i<n;i++){
            deg[i]=in[i]+out[i];
        }
        ll maxD=*max_element(all(deg));
        vlld degDis(maxD+1),cdegDis(maxD+1);
        for(auto x:deg){
            mpd[x]++;
        }

        for(int i=0;i<=maxD;i++){
            if(mpd.count(i))
                degDis[i]=mpd[i];
        }
        cdegDis[0]=degDis[0];
        for(int i=1;i<=maxD;i++){
            cdegDis[i]=cdegDis[i-1]+degDis[i];
        }
        cout<<"Degree distribution"<<endl;
        for(int i=1;i<=maxD;i++){
            cout<<degDis[i]<<" ";
        }
        cout<<endl;
        cout<<"Cummulative Degree distribution"<<endl;
        for(int i=1;i<=maxD;i++){
            cout<<cdegDis[i]<<" ";
        }
        cout<<endl;


    }
    
    void localClusteringCoefficient(){
        vlld lcc(n);
        for(int node=0;node<n;node++){
            auto temp=this->adjList[node];
            ll k=temp.size(),ans=0;
            for(int i=0;i<k;i++){
                for(int j=i+1;j<k;j++){
                    if(this->adjMat[temp[i]][temp[j]]){
                    ans++;
                    }
                }
            }    

            lcc[node]=((lld)2*ans)/(k*(k-1));
        }
        cout<<"**** Local Clustering Coeficients ****"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={lcc[i],i};
            // cout<<lcc[i]<<" ";
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
        

    }
    
    void globalClusteringCoefficient(){
        ll closedTriangles=0,openTriangles=0;
        vvll temp=this->adjMat;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(temp[i][j]){
                    for(int k=0;k<n;k++){
                        if(k!=i && k!=j){
                            if((temp[i][k]==0 && temp[j][k]==1) || (temp[i][k]==1 && temp[j][k]==0)){
                                openTriangles++;
                            }
                            else if(temp[i][k]==1 && temp[j][k]==1){
                                closedTriangles++;
                            }
                        }
                    }
                }
                
            }
        }
        cout<<"Closed Triangles : "<<closedTriangles<<" || Open Triangles : "<<" "<<openTriangles<<endl;
        lld temp1=((lld)closedTriangles/(closedTriangles+openTriangles));
        cout<<"Global Clutering Coefficient"<<endl;
        cout<<temp1<<endl;
    }

    void bfs(ll src,vll& vis){
        
        queue<ll> q;q.push(src);
    
        while(!q.empty()){
            ll u=q.front();q.pop();
            for(auto v:this->adjList[u]){
                if(vis[v]==0){
                    vis[v]=1;
                    q.push(v);
                }
            }
        }
        return ;
    
    }        
    
    void connectedComponents(){
        vll vis(n);
        ll count=0;
        for(int i=0;i<n;i++){
            if(vis[i]==0){
                count++;
                bfs(i,vis);
            }               
        }
        cout<<"No. of connected Components"<<endl;
        cout<<count<<endl;
    }
    
    void degreeCentrality(){
        vlld ans(n);
        vll in(n),out(n),deg(n);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(adjMat[i][j]){
                out[i]++;
                in[j]++;
                }
            }
        }
        for(int i=0;i<n;i++){
            deg[i]=in[i]+out[i];
        }
        ll maxD=*max_element(all(deg));
        for(int i=0;i<n;i++){
            ans[i]=(lld)deg[i]/maxD;
        }

        cout<<"**** Degree Centrality ****"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={ans[i],i};
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
        
        
    }
    
    void closenessCentrality(){ //floydWarshall

        vlld ans(n);
        
        for(int i=0;i<n;i++){
            d[i][i]=0;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(adjMat[i][j])d[i][j]=1;
            }
        }
        for(int k=0;k<n;k++){
            for(int i=0;i<n;i++){
                for(int j=0;j<n;j++){
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
                }
            }
        }

        for(int i=0;i<n;i++){
            auto temp=accumulate(all(d[i]),0ll);
            ans[i]=(lld)(n-1)/temp;
        }
        // display("Closeness Centrality",ans,1,1);
        cout<<"**** Closeness Centrality ****"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={ans[i],i};
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
    }
    
    void eigenVectorCentrality(ll show=0,ll rank=5){ //power iteration method
        cout<<"begin of Eigen vector"<<endl;
        lld normalizedValue;
        vvlld ans(n,vlld(1,1)),adjm;
        adjm=matOps.copy(adjMat);
        ll itr=7;
        while(itr--){
            lld temp=0;
            ans=matOps.mul(adjm,ans);
            cout<<"after mat mul"<<endl;
            for(int i=0;i<n;i++){
                temp+=ans[i][0]*ans[i][0];
            }
            normalizedValue=(lld)sqrt(temp);
            for(int i=0;i<n;i++){
                ans[i][0]/=normalizedValue;
            }
            

        }
        cout<<"Eigen Vector Centrality"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={ans[i][0],i};
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
        if(show){
            for(int i=0;i<rank;i++){
                cout<<"***************"<<endl;;
                cout<<"Rank  : "<<i+1<<endl;
                ll paper=rmp[a[i].second];
                publication(paper);
                cout<<endl;
            
            }
        }
        
    }

    void diameter(){
        closenessCentrality();
        ll mx=LLONG_MIN;
        for(auto v:d){
            for(auto x:v){
                mx=max(mx,x);
            }
        }
        cout<<"Diameter of Network is : "<<mx<<endl;
    }

    vvll bfsDis(ll src){
        vll vis(n),temp;
        vvll res;
        queue<int> q;q.push(src);q.push(-1);
        res.pb({});
        vis[src]=1;
        while(!q.empty()){
            ll u=q.front();q.pop();
            for(auto v:adjList[u]){
                if(vis[v]==0){
                    temp.pb(v);
                    vis[v]=1;
                    q.push(v);
                }
            }
            if(q.front()==-1){
                q.pop();
                if(!q.empty()){
                    res.pb(temp);
                    temp={};
                    q.push(-1);
                }
                
            }
        }
        return res;

    }
    void katzCentrality(ll show=0,ll rank=5){ 
        vvvlld A(10,vvlld(n,vlld(n,0)));
        vlld ans(n);
        for(int i=0;i<n;i++){
            // cout<<"NODE-->"<<i<<endl;
            vvll res=bfsDis(i);
            // cout<<"{ ";
            // for(int i=0;i<res.size();i++){
            //     cout<<"{";
            //    for(int j=0;j<res[i].size();j++){
            //       cout<<res[i][j]<<",";
            //    }
            //    cout<<"}";
            // }
            // cout<<" }"<<endl;
            for(int a=1;a<min(10,(int)res.size());a++){
                for(int b=0;b<res[a].size();b++){
                    A[a][i][res[a][b]]=1;
                }
            }
        }

        for(int i=0;i<n;i++){
            for(int k=0;k<A.size();k++){
                ll temp=0;
                lld alpha=0.2;
                for(int j=0;j<n;j++){
                    temp+=A[k][j][i];
                }
                ans[i]+=temp*pow(alpha,k+1);
            }
        }

        cout<<"Katz Centrality"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={ans[i],i};
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
        
        if(show){
            for(int i=0;i<rank;i++){
                cout<<"***************"<<endl;;
                cout<<"Rank  : "<<i+1<<endl;
                ll paper=rmp[a[i].second];
                publication(paper);
                cout<<endl;
            
            }
        }
        

    }

    void pageRank(ll show=0,ll rank=5,ll iterations=7){
        vvlld R(1,vlld(n,(lld)1/n)),E(1,vlld(n,(lld)1/n)),A(n,vlld(n,0));
        lld d=0.85;
        E=matOps.scalarMul(1-d,E);
        
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                A[i][j]=adjMat[i][j];
            }
        }
            
        for(int i=0;i<iterations;i++){
            R=matOps.mul(R,A);
            R=matOps.scalarMul(d,R);
            R=matOps.add(E,R);       
        }
        cout<<"***** Page Rank ****"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={R[0][i],i};
        }
        sort(rall(a));
        for(int i=0;i<R[0].size();i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
        
        
        if(show){
            for(int i=0;i<rank;i++){
                cout<<"***************"<<endl;;
                cout<<"Rank  : "<<i+1<<endl;
                ll paper=rmp[a[i].second];
                publication(paper);
                cout<<endl;
            
            }
        }
    }

    void calculateDensity() {
        if (n < 2) cout<<"Density of Network is : "<< 0.0<< endl;
        cout<<"Density of Network is : "<< (2.0 * m) / (n * (n - 1)) << endl;
    }

    void calculateHubAuth(int maxIterations = 100, lld tol = 1e-5) {
        vlld hub(n, 1.0);      
        vlld auth(n, 1.0);     
        vlld newHub(n, 0.0);   
        vlld newAuth(n, 0.0);   

        for (int iter = 0; iter < maxIterations; iter++) {
            for (int i = 0; i < n; i++) {
                newAuth[i] = 0.0;
                for (int j = 0; j < n; j++) {
                    if (adjMat[j][i] == 1) {
                        newAuth[i] += hub[j];  // Authority is the sum of hubs of incoming nodes
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                newHub[i] = 0.0;
                for (int j = 0; j < n; j++) {
                    if (adjMat[i][j] == 1) {
                        newHub[i] += auth[j];  // Hub is the sum of authorities of outgoing nodes
                    }
                }
            }

            lld maxAuth = *max_element(all(newAuth));
            lld maxHub = *max_element(all(newHub));

            for (int i = 0; i < n; i++) {
                auth[i] = newAuth[i] / maxAuth;
                hub[i] = newHub[i] / maxHub;
            }

            lld hubDiff = 0.0, authDiff = 0.0;
            for (int i = 0; i < n; i++) {
                hubDiff += fabs(hub[i] - newHub[i]);
                authDiff += fabs(auth[i] - newAuth[i]);
            }

            if (hubDiff < tol && authDiff < tol) {
                break; 
            }

        }

        // // Output the final hub and authority scores
        // display("Hub",hub);
        // display("Auth",auth);

        
        cout<<"***** Hub ****"<<endl;
        vpldll a(n);
        for(int i=0;i<n;i++){
            a[i]={hub[i],i};
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }


        cout<<"***** Auth ****"<<endl;
        for(int i=0;i<n;i++){
            a[i]={auth[i],i};
        }
        sort(rall(a));
        for(int i=0;i<n;i++){
            cout<<rmp[a[i].second]<<" --> "<<a[i].first<<endl;
        }
        
        
    }
    
    int countBidirectionalEdges() {
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (adjMat[i][j] == 1 && adjMat[j][i] == 1) {
                    count++;
                }
            }
        }
        return count;
    }

    void calculateReciprocity() {
        ll totalDirectedEdges = m;
        if (totalDirectedEdges == 0) cout<<"Reciprocity  :  "<<0.0<<endl;; 

        ll bidirectionalEdges = countBidirectionalEdges();
        auto temp = (2.0 * bidirectionalEdges) / totalDirectedEdges; 
        cout<<"Reciprocity  :  "<<temp<<endl;
    }

    void linkPredClusteringCoeff(ll A,ll B){
        lld count=0;
        set<ll> a;
        for(auto x:adjList[A]){
            a.insert(x);
        }
        for(auto x:adjList[B]){
            if(a.count(x))count++;
        }
        for(auto x:adjList[B]){
            a.insert(x);
        }

        auto ans= count/(lld)(a.size());
        cout<<"Link prediction for "<<A<<" & "<<B<<" ->"<<ans<<endl;
        
    }
    
    
    lld f(ll A, ll B, lld C, int maxDepth, map<pll, lld>& memo) {
        if (A == B) return 1.0;
        if (maxDepth == 0 || adjList[A].empty() || adjList[B].empty()) return 0.0;
        if (memo.find({A, B}) != memo.end()) {
            return memo[{A, B}];
        }

        vll neighborsA = adjList[A], neighborsB = adjList[B];
        lld simSum = 0.0;
        for (int neighborA : neighborsA) {
            for (int neighborB : neighborsB) {
                simSum += f(neighborA, neighborB, C, maxDepth - 1, memo);
            }
        }

        lld similarity = (C * simSum) / (neighborsA.size() * neighborsB.size());

        memo[{A, B}] = similarity;

        return similarity;
    }
    void simRank(ll A,ll B){
        lld C=0.8,rank=0;
        int maxDepth = 10; // Maximum recursion depth
        map<pll, lld> memo;
        rank = f(A,B,C,maxDepth,memo);
        cout<<"Sim Rank for "<<A<<" "<<B<<" "<<rank<<endl;
    }

    
    void cosineSim(vvll t){
        vll msim(t.size());
        for (int i = 0; i < t.size(); i++){
            cout<<i<<endl;
            lld ans = -1e9, mostlikely = i;
            for (int j = 0; j < t.size(); j++){
            
                if (i != j)            {
                    lld l = 0, r = 0, lr = 0;

                    for (int k = 0; k < t[0].size(); k++){
                        l += t[i][k] * t[i][k];
                        r += t[j][k] * t[j][k];
                        lr += t[i][k] * t[j][k];
                    }
                    lld temp = (lr) / sqrt(l * r);
                    cout<<j<<" --> "<<temp<<endl;
                    if (temp > ans){
                        ans = temp;
                        mostlikely = j;
                    }
                }
            }
            msim[i] = mostlikely;
        }
        cout<<"******************"<<endl;
        for (int i = 0; i < t.size(); i++)    {
            cout << "Most similiar to " << i << " is " << msim[i] << endl;
        }
    }
    void cosineSim(){
        vvll t={
            {2,1,0,0},
            {50,20,0,0},
            {2,0,1,0},
            {2,1,0,0},
            {0,0,1,1}
        };
        cosineSim(t);
    }

    void pathSim(vvll t){
        vll msim(t.size());
        for (int i = 0; i < t.size(); i++){
            cout<<i<<endl;
            lld ans = -1e9, mostlikely = i;
            for (int j = 0; j < t.size(); j++){
            
                if (i != j)            {
                    lld l = 0, r = 0, lr = 0;

                    for (int k = 0; k < t[0].size(); k++){
                        l += t[i][k] * t[i][k];
                        r += t[j][k] * t[j][k];
                        lr += t[i][k] * t[j][k];
                    }
                    lld temp = (lr) / (l + r);
                    cout<<j<<" --> "<<temp<<endl;
                    if (temp > ans){
                        ans = temp;
                        mostlikely = j;
                    }
                }
            }
            msim[i] = mostlikely;
        }
        cout<<"******************"<<endl;
        for (int i = 0; i < t.size(); i++)    {
            cout << "Most similiar to " << i << " is " << msim[i] << endl;
        }
    }
    void pathSim(){
        vvll t={
            {2,1,0,0},
            {50,20,0,0},
            {2,0,1,0},
            {2,1,0,0},
            {0,0,1,1}
        };
        pathSim(t);
    }

    void modularity(vvll& comm, vvll& aL, vvll& aM){
        lld q=0,m=0;
        for(auto v:aL){
            m+=v.size();
        }
        m/=2;
        for(int i=0;i<comm.size();i++){
            lld mn=0,kn=0;
            auto commn=comm[i];
            for(int j=0;j<commn.size();j++){
                kn+=aL[commn[j]].size();
                for(int k=j+1;k<commn.size();k++){
                    if(aM[commn[j]][commn[k]]){
                        mn++;
                    }
                }
            }
            q+=((mn/m)-(kn/(2*m))*(kn/(2*m)));
            
        }

        cout<<"Modularity for given community : "<<q<<endl;

    }
    void modularity(){
        vvll comm={{0,1,2,3},{4,5,6,7,8,9}};
        vvll edN={{0,1},{0,3},{1,2},{3,2},{2,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,4}};
        vvll aL(10),aM(10,vll(10,0));
        for(auto x:edN){
            aL[x[0]].push_back(x[1]);
            aL[x[1]].push_back(x[0]);
            aM[x[0]][x[1]]=1;
            aM[x[1]][x[0]]=1;
        }
        modularity(comm,aL,aM);
        
    }
    
    void purity(ll V,vvll detected, vvll groundTruth){
        lld ans=0;
        for(int i=0;i<detected.size();i++){
            auto temp=detected[i];
            ll cnt=0;
            for(auto t:groundTruth){
                ll intersection=0;
                set<ll> st,stemp;;
                ll j=0;
                for(auto x:t){
                    st.insert(x);
                }
                for(auto x:temp){
                    stemp.insert(x);
                }

                for(auto x:st){
                    if(stemp.count(x))j++;
                }
                cnt=max(cnt,j);

            }
            ans+=cnt;
        }

        cout<<"Purity for this detected community"<<endl;
        cout<<(lld)ans/V<<endl;
        

    }
    void purity(){
        ll V=7;
        vvll d={{1,1,1,1,2},{2,2,3,3,3},{2,2,2,2,2,3}};
        vvll g={{1,1,1,1},{2,2,2,2,2,2,2,2,2},{3,3,3,3}};
        purity(V,d,g);
    }

    void omegaIndex(ll V,vvll detected, vvll groundTruth){
        lld ans=0;
        vvll l(V,vll(V,0)),r(V,vll(V,0));
        for(int i=0;i<detected.size();i++){
            auto temp=detected[i];
            for(int j=0;j<temp.size();j++){
                for(int k=0;k<temp.size();k++){
                    l[temp[j]][temp[k]]++;
                }
            }
        }
        for(int i=0;i<groundTruth.size();i++){
            auto temp=groundTruth[i];
            for(int j=0;j<temp.size();j++){
                for(int k=0;k<temp.size();k++){
                    r[temp[j]][temp[k]]++;
                }
            }
        }
        for(int i=0;i<V;i++){
            for(int j=0;j<V;j++){
                if(l[i][j]==r[i][j])ans++;
            }
        }
        // debug(l);
        // debug(r);
        cout<<ans/(V*V)<<endl;
        

    }
    void omegaIndex(){
        ll V=7;
        vvll d={{0,1,2,6},{1,3,5},{3,4,5,6}};
        vvll g={{0,1,2,3,6},{0,1,3,4,5},{4,5,6}};
        omegaIndex(V,d,g);
    }

    lld internalClusteringCoefficient(ll node, vll &communityNodes,  map<ll, vector<ll>> &graph){
        unordered_set<ll> communitySet(communityNodes.begin(), communityNodes.end());

        vector<ll> internalNeighbors;
        for (ll neighbor : graph.at(node)){
            if (communitySet.find(neighbor) != communitySet.end()){
                internalNeighbors.push_back(neighbor);
            }
        }

        ll internalEdges = 0, possibleEdges = 0;
        ll size = internalNeighbors.size();

        for (int i = 0; i < internalNeighbors.size(); i++){
            for (int j = i + 1; j < internalNeighbors.size(); j++){
                ll u = internalNeighbors[i];
                ll v = internalNeighbors[j];
                if (find(graph.at(u).begin(), graph.at(u).end(), v) != graph.at(u).end()){
                    internalEdges++;
                }
            }
        }

        possibleEdges = size * (size - 1) / 2;
        return possibleEdges > 0 ? static_cast<lld>(internalEdges) / possibleEdges : 0.0;
    }
    lld calculatePermanence(ll node,vll &communityNodes,map<ll, vll> &graph,map<ll, ll> &nodeToCommunity){
        ll internalNeighbors = 0;
        map<ll, ll> externalConnections;

        for (ll neighbor : graph.at(node)){
            if (find(communityNodes.begin(), communityNodes.end(), neighbor) != communityNodes.end()) {
                internalNeighbors++;
            }
            else{
                externalConnections[nodeToCommunity.at(neighbor)]++;
            }
        }

        ll maxExternalConnections = 0;
        for (const auto &[community, connections] : externalConnections){
            maxExternalConnections = max(maxExternalConnections, connections);
        }

        ll degree = graph.at(node).size();
        if (degree == 0)
            return 0.0;

        lld cin = internalClusteringCoefficient(node, communityNodes, graph);

        if (maxExternalConnections == 0)
            return 0.0;

        lld permanence = (static_cast<lld>(internalNeighbors) / maxExternalConnections) * (1.0 / degree) - 1.0 + cin;

        cout << "Node: " << node << endl;
        cout << "Internal Neighbors (I_v): " << internalNeighbors << endl;
        cout << "Max External Connections (E_max): " << maxExternalConnections << endl;
        cout << "Degree (deg_v): " << degree << endl;
        cout << "Internal Clustering Coefficient (c_in): " << cin << endl;
        cout << "Permanence: " << permanence << endl;

        return permanence;
    }
    void permanence(){
            map<ll, vector<ll>> graph = {
            {0, {1, 4, 5}},
            {1, {0, 4, 2, 3}},
            {2, {1, 4, 3}},
            {3, {1, 2, 5}},
            {4, {0, 1, 2, 5}},
            {5, {0, 3, 4}}};

        map<ll, ll> nodeToCommunity = {
            {0, 1}, {1, 1}, {2, 1}, {4, 1}, {3, 2}, {5, 2}};

        map<ll, vll> communities = {
            {1, {0, 1, 2, 4}},
            {2, {3, 5}}};

        ll node = 4;
        ll communityId = nodeToCommunity[node];

        if (communities.find(communityId) != communities.end()){
            lld permanence = calculatePermanence(node, communities[communityId], graph, nodeToCommunity);
            cout << "Permanence of node " << node << ": " << permanence << endl;
        }
        else{
            cout << "Node does not belong to a valid community!" << endl;
        }
    }

    bool isClique(vvll& adj, const vll& nodes) {
        for (int i = 0; i < nodes.size(); ++i) {
            for (int j = i + 1; j < nodes.size(); ++j) {
                if (find(adj[nodes[i]].begin(), adj[nodes[i]].end(), nodes[j]) == adj[nodes[i]].end()) {
                    return false;
                }
            }
        }
        return true;
    }
    void findCliques(vvll& adj, vvll& cliques, vll& temp, ll start, ll k) {
        if (temp.size() == k) {
            if (isClique(adj, temp)) {
                cliques.push_back(temp);
            }
            return;
        }

        for (int i = start; i < adj.size(); ++i) {
            temp.push_back(i);
            findCliques(adj, cliques, temp, i + 1, k);
            temp.pop_back();
        }
    }
    vvll buildCommunities(vvll& cliques, ll k) {
        vvll communities;
        vll visited(cliques.size(), 0);

        function<void(ll, vll&)> dfs = [&](ll idx, vll& community) {
            visited[idx] = 1;
            community.insert(community.end(), cliques[idx].begin(), cliques[idx].end());

            for (int i = 0; i < cliques.size(); ++i) {
                if (!visited[i]) {
                    ll commonNodes = 0;
                    for (ll node : cliques[idx]) {
                        if (find(cliques[i].begin(), cliques[i].end(), node) != cliques[i].end()) {
                            ++commonNodes;
                        }
                    }
                    if (commonNodes >= k - 1) {
                        dfs(i, community);
                    }
                }
            }
        };

        for (int i = 0; i < cliques.size(); ++i) {
            if (!visited[i]) {
                vll community;
                dfs(i, community);
                sort(community.begin(), community.end());
                community.erase(unique(community.begin(), community.end()), community.end());
                communities.push_back(community);
            }
        }
        return communities;
    }
    void cliquePercolation() {
        ll n = 7; 
        vvll adj(n);
        
        
        vpll edges = {
            {0, 1}, {0, 2}, {1, 2},  // Triangle between 0, 1, 2
            {2, 3}, {3, 4}, {2, 4},  // Triangle between 2, 3, 4
            {4, 5}, {5, 6}, {4, 6},  // Triangle between 4, 5, 6
            {1, 5}
        };

        for (auto& edge : edges) {
            adj[edge.first].push_back(edge.second);
            adj[edge.second].push_back(edge.first);
        }

        ll k = 3; 
        vvll cliques;
        vll temp;

        findCliques(adj, cliques, temp, 0, k);

        cout << "Found " << k << "-cliques:"<<endl;;
        for (auto& clique : cliques) {
            for (ll node : clique) {
                cout << node << " ";
            }
            cout <<endl;
        }

        vvll communities = buildCommunities(cliques, k);

        cout << "\nDetected Communities:\n";
        for (int i = 0; i < communities.size(); ++i) {
            cout << "Community " << i + 1 << ": ";
            for (ll node : communities[i]) {
                cout << node << " ";
            }
            cout <<endl;
        }

    }


};

