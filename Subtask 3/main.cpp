#include<iostream>
#include "audio.h"
#include<vector>
#include<fstream>

using namespace std;

typedef struct pred_t
{
    int label;
    float prob;
}pred_t;

int main(int argc, char** argv)
{
    vector<string> keywords;
    keywords.push_back("silence");
    keywords.push_back("unknown");
    keywords.push_back("yes");
    keywords.push_back("no");
    keywords.push_back("up");
    keywords.push_back("down");
    keywords.push_back("left");
    keywords.push_back("right");
    keywords.push_back("on");
    keywords.push_back("off");
    keywords.push_back("stop");
    keywords.push_back("go");
    try
    {
        if(argc == 1)
        {
            throw invalid_argument("No arguments provided!");
        }
        else if(argc == 2)
        {
            throw invalid_argument("Output file not specified!");
        }
        else if(argc == 3)
        {
            pred_t* pred;
            pred = (pred_t*)malloc(3 * sizeof(pred_t));
            // libaudioAPI takes a filename and an array of elements pred_t, and returns another array of pred_t which contains the 
            // information about the maximum probabilities and their respective keywords
            pred = libaudioAPI(argv[1], pred);
            ofstream outdata;
            outdata.open(argv[2], std::ios_base::app);
            outdata<<argv[1]<<" "<<keywords[pred[0].label]<<" "<<keywords[pred[1].label]<<" "<<keywords[pred[2].label]<<" "<<pred[0].prob<<" "<<pred[1].prob<<" "<<pred[2].prob<<endl;
            outdata.close();
        }
        else
        {
            throw invalid_argument("Too many command line arguments!");
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}