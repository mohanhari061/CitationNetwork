#include <bits/stdc++.h>
using namespace std;
namespace fs = filesystem;
#define ll long long
class FindAbs
{
public:
    map<string, string> m;
    
    string year, paper;

    FindAbs(string paper, string year){
        this->paper=paper;
        this->year=year;
        publication();
    }

    void convert(string input)
    {
        stringstream ss(input);
        string line;
        string abstract;
        bool inAbstract = false;

        while (std::getline(ss, line))
        {
            // Ignore separator lines
            if (line == "------------------------------------------------------------------------------" || line == "\\")
            {
                continue;
            }

            // Check for key-value pairs
            size_t pos = line.find(": ");
            if (pos != std::string::npos)
            {
                // Extract the key and value
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2);
                m[key] = value;
            }
            else if (line.find("\\\\") != std::string::npos)
            {
                // Start capturing the abstract section after the last '\\'
                inAbstract = true;
            }
            else if (inAbstract)
            {
                // Collect the abstract content
                if (line.find("\\\\") == std::string::npos)
                {
                    abstract += line + " ";
                }
            }
        }

        // Add abstract to key-value pairs
        if (!abstract.empty())
        {
            m["Abstract"] = abstract;
        }
    }

    void publication()
    {
        string filePath = "../Dataset/cit-HepTh-abstracts/" + year + "/" + paper + ".abs";
        ifstream file(filePath);

        string content = "";
        if (file.is_open())
        {
            string line;
            while (getline(file, line))
            {
                content += line + "\n";
            }
            file.close();
            convert(content);
        }
        else
        {
            cout << "Not Found\n";
        }
    }
}
;
