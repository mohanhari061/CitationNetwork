#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iomanip>
using namespace std;

struct pair_hash {
    template <typename T1, typename T2>
    std::size_t operator ()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);  // Combine the two hash values using XOR and bit-shift
    }
};
double internalClusteringCoefficient(int node, const vector<int>& communityNodes, const unordered_map<int, vector<int>>& graph) {
    unordered_set<int> communitySet(communityNodes.begin(), communityNodes.end());

    // Get neighbors of the node that are inside the community
    vector<int> internalNeighbors;
    for (int neighbor : graph.at(node)) {
        if (communitySet.find(neighbor) != communitySet.end()) {
            internalNeighbors.push_back(neighbor);
        }
    }

    int internalEdges = 0, possibleEdges = 0;
    int size = internalNeighbors.size();

    // Count actual internal edges between neighbors
    for (size_t i = 0; i < internalNeighbors.size(); i++) {
        for (size_t j = i + 1; j < internalNeighbors.size(); j++) {
            int u = internalNeighbors[i];
            int v = internalNeighbors[j];
            if (find(graph.at(u).begin(), graph.at(u).end(), v) != graph.at(u).end()) {
                internalEdges++;
            }
        }
    }

    // Calculate possible edges between internal neighbors
    possibleEdges = size * (size - 1) / 2;

    return possibleEdges > 0 ? static_cast<double>(internalEdges) / possibleEdges : 0.0;
}

// Function to calculate the x_e value for each edge
unordered_map<pair<int, int>, int, pair_hash> calculateEdgeCommunityCount(
    const unordered_map<int, vector<int>>& graph,
    const unordered_map<int, vector<int>>& communities
) {
    unordered_map<pair<int, int>, int, pair_hash> edgeCommunityCount;

    // Iterate over all communities
    for (const auto& community : communities) {
        int communityId = community.first;
        const vector<int>& nodes = community.second;
        unordered_set<int> nodeSet(nodes.begin(), nodes.end());

        // Count edges within this community
        for (int u : nodes) {
            for (int v : graph.at(u)) {
                if (nodeSet.find(v) != nodeSet.end()) {
                    pair<int, int> edge = minmax(u, v); // Store edges as unordered pairs
                    edgeCommunityCount[edge]++;
                }
            }
        }
    }

    return edgeCommunityCount;
}

// Function to calculate Generalized Permanence
double calculateGeneralizedPermanence(
    int node,
    const vector<int>& communityNodes,
    const unordered_map<int, vector<int>>& graph,
    const unordered_map<int, int>& nodeToCommunity,
    const unordered_map<pair<int, int>, int, pair_hash>& edgeCommunityCount
) {
    int internalNeighbors = 0;
    unordered_map<int, int> externalConnections;

    // Count internal and external neighbors
    for (int neighbor : graph.at(node)) {
        if (find(communityNodes.begin(), communityNodes.end(), neighbor) != communityNodes.end()) {
            internalNeighbors++;
        } else {
            externalConnections[nodeToCommunity.at(neighbor)]++;
        }
    }

    int maxExternalConnections = 0;
    for (const auto& externalConnection : externalConnections) {
        maxExternalConnections = max(maxExternalConnections, externalConnection.second);
    }

    int degree = graph.at(node).size();
    if (degree == 0) return 0.0;

    double cin = internalClusteringCoefficient(node, communityNodes, graph);

    // Calculate the edge contribution term
    double edgeContribution = 0.0;
    for (int neighbor : communityNodes) {
        if (neighbor != node) {
            pair<int, int> edge = minmax(node, neighbor);
            if (edgeCommunityCount.find(edge) != edgeCommunityCount.end()) {
                edgeContribution += 1.0 / edgeCommunityCount.at(edge);
            }
        }
    }

    // Generalized Permanence formula
    double permanence = (static_cast<double>(internalNeighbors) / maxExternalConnections) * (1.0 / degree)
                        - (1.0 - cin) * (edgeContribution / internalNeighbors);

    return permanence;
}

int main() {
    // Example graph represented as an adjacency list
    unordered_map<int, vector<int>> graph = {
        {0, {1, 4, 5}},
        {1, {0, 4, 2, 3}},
        {2, {1, 4, 3}},
        {3, {1, 2, 5}},
        {4, {0, 1, 2, 5}},
        {5, {0, 3, 4}}
    };

    // Node to community mapping
    unordered_map<int, int> nodeToCommunity = {
        {0, 1}, {1, 1}, {2, 2}, {4, 2},
        {3, 2}, {5, 2}
    };

    // Community definitions
    unordered_map<int, vector<int>> communities = {
        {1, {0, 1}},
        {2, {2, 4, 3, 5}}
    };

    // Calculate edge community counts
    auto edgeCommunityCount = calculateEdgeCommunityCount(graph, communities);

    int node = 2; // Node for which generalized permanence is to be calculated
    int communityId = nodeToCommunity[node];

    if (communities.find(communityId) != communities.end()) {
        double permanence = calculateGeneralizedPermanence(node, communities[communityId], graph, nodeToCommunity, edgeCommunityCount);
        cout << "Generalized Permanence of node " << node << ": " << permanence << endl;
    } else {
        cout << "Node does not belong to a valid community!" << endl;
    }

    return 0;
}