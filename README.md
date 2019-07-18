# KMeans
* Single-threaded KMeans based on C++11.
* KMeans++ initialization. 
* Only include single-header file.

#include <ctime>
#include <cstring>

#include "kmeans.h"

int main()
{
	srand((unsigned)time(NULL));
	
	KMeans kmeans;
	
	int n_clusters = 3; // the number of cluster centers
	std::vector<int> labels; // cluster labels
	std::string init_method = "kmeans++"; // support "kmeans++" and "random" initialization.
	std::vector< std::vector<float> > features; // 2D matrix, [n_samples, n_features]
	
	labels = kmeans.fit_predict(features, n_clusters, init_method);
	
	
	return 0;
}
