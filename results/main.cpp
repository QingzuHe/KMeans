#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>

#include <vector>

#include "kmeans.h"

std::string name = "X";

void LoadData(std::vector< std::vector<float> > &features)
{
	std::string file_name = name+".txt";
	
	std::ifstream DataFile(file_name.c_str());
	
	if (!DataFile)
	{
		printf("Can not open data file. No such file or directory");
		exit(0);
	}
	
	std::string line;
	
	while (std::getline(DataFile, line))
	{
		if (line.empty()) continue;
		
		int pos = line.find(" ");
		
		float x = atof(line.substr(0, pos).c_str());
		float y = atof(line.substr(pos+1).c_str());
		
		std::vector<float> tmp = {x, y};
		
		features.push_back(tmp);
	}
	
	DataFile.close();
	
}

void OutputLabels(std::vector<int> &labels)
{
	std::string file_name = name+"_labels.txt";
	
	std::ofstream LabelsFile(file_name.c_str());
	
	if (!LabelsFile)
	{
		printf("Can not open output file.");
		exit(0);
	}
	
	for (int i = 0; i < labels.size(); ++i)
	{
		LabelsFile << std::to_string(labels[i]) << "\n";
	}
	
	LabelsFile.close();
}

int main()
{
	srand((unsigned)time(NULL));
	
	KMeans kmeans;
	
	int n_clusters = 3; // the number of cluster centers
	std::vector<int> labels; // cluster labels
	std::string init_method = "kmeans++"; // support "kmeans++" and "random" initialization.
	std::vector< std::vector<float> > features; // 2D matrix, [n_samples, n_features]
	
	LoadData(features);
	
	labels = kmeans.fit_predict(features, n_clusters, init_method);
	
	OutputLabels(labels);
	
	return 0;
}