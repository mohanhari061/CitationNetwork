#include "../Graph/graph.cpp" 
#include <bits/stdc++.h>

using namespace std;

int main(){
    ios_base::sync_with_stdio(false); 
    cin.tie(NULL);

    cout << "Enter number of nodes, edges, and if it's directed (0 for undirected, 1 for directed),Enter 1 to input edges manually, or 2 to read from file: " << endl;
    ll n, m, directed, choice = -1; 
    cin >> n >> m >> directed;
    vvll edges(m); 
    map<ll, ll> mp, rmp;  
    ll temp = 0;  

    cout << "Enter 1 to input edges manually, or 2 to read from file: ";
    ll inputChoice;
    cin >> inputChoice;

    if(inputChoice == 1) {
        cout << "Enter the edges (u v):" << endl;
        for(int i = 0; i < m; i++){
            ll u, v;
            cin >> u >> v;
            
            if(mp.count(u) == 0){
                mp[u] = temp;
                rmp[temp] = u;
                temp++;
            }
            if(mp.count(v) == 0){
                mp[v] = temp;
                rmp[temp] = v;
                temp++;
            }
            edges[i] = {mp[u], mp[v]};
        }
    } else if(inputChoice == 2) {
        string filename="H:/MCA/3rd Sem/3rdSem/SNA/CitationNetwork/Dataset/Sample/1000nodes/edges.txt";
        

        ifstream infile(filename);
        if (!infile) {
            cerr << "Error: File not found!" << endl;
            return 1;
        }

        ll u, v;
        int i = 0;
        while (infile >> u >> v) {
            if(i >= m) break;  // Ensure we don't exceed expected number of edges
            
            if(mp.count(u) == 0){
                mp[u] = temp;
                rmp[temp] = u;
                temp++;
            }
            if(mp.count(v) == 0){
                mp[v] = temp;
                rmp[temp] = v;
                temp++;
            }
            edges[i++] = {mp[u], mp[v]};
        }
        infile.close();
    } else {
        cout << "Invalid choice!" << endl;
        return 1;
    }

    graph g(n, edges, rmp, directed);

    do {
        cout <<endl<< "--- Graph Metrics Menu ---"<<endl;
        cout << "1. Degree Distribution"<<endl;
        cout << "2. Diameter"<<endl;
        cout << "3. LCC (Local Clustering Coefficient)"<<endl;
        cout << "4. GCC (Global Clustering Coefficient)"<<endl;
        cout << "5. Connected Components"<<endl;
        cout << "6. Degree Centrality"<<endl;
        cout << "7. EigenVector Centrality"<<endl;
        cout << "8. Katz Centrality"<<endl;
        cout << "9. Closeness Centrality"<<endl;
        cout << "10. Hub & Authority"<<endl;
        cout << "11. PageRank"<<endl;
        cout << "12. Reciprocity"<<endl;
        cout << "13. Display Network"<<endl;
        cout << "Otherwise. Exit"<<endl;
        cout << "Enter your choice: --> ";
        cin >> choice;
       
        switch(choice) {
            case 1: g.degreeDistribution(); break;
            case 2: g.diameter(); break;
            case 3: g.localClusteringCoefficient(); break;
            case 4: g.globalClusteringCoefficient(); break;
            case 5: g.connectedComponents(); break;
            case 6: g.degreeCentrality(); break;
            case 7: g.eigenVectorCentrality(1, 7); break;
            case 8: g.katzCentrality(1, 8); break;
            case 9: g.closenessCentrality(); break;
            case 10: g.calculateHubAuth(); break;
            case 11: g.pageRank(1, 11); break;
            case 12: g.calculateReciprocity(); break;
            case 13: g.display(); break;
            case 14: g.permanence(); break;
            case 0: cout << "Exiting..."<<endl; break;
            default: cout << "Invalid choice, please try again."<<endl; break;
        }
    } while (choice != 0);

    return 0;
}
