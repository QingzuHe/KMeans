# KMeans
* Single-threaded KMeans based on C++11.
* KMeans++ and random initialization. 
* Only include single-header file.
* Euclidean distance.

## Usage
### Modification
Before you put "kmeans.h" to your project, you need modify 11th line in this file.
```c++
#define N_FEATURES 2
```
N_FEATURES means the number of features. For example, you have 100 samples and each sample includes 2 dimensions, so N_FEATURES=2. We define this parameter for accelerate computation.

### Make
```shell
g++ -O3 -std=c++11 -o kmeans main.cpp
```

## Example
```c++
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
```

## In the future
We will try to implement multi-threads KMeans algorithm in CPU. However, we will use thread-safe queue to replace "omp.h". 

## Comparison with sklearn.cluster.KMeans(popular python packge)
![image](https://github.com/QingzuHe/KMeans/raw/master/results/ResultsOfComparison.jpg)

