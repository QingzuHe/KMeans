#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <algorithm>

#define MAX_ITER 200
#define FLT_MAX 3.402823466e+38F 

#define N_INIT 10
#define N_FEATURES 2

typedef struct 
{
	int group = 0; 
	float clu_dist;
	float data[N_FEATURES];
} Point;

class KMeans
{
	private:
		
		float distance(Point &x, Point &y);
		
		int nearest(Point &pt, std::vector<Point> &centers, int n_centers, float *d2);
		
		void kpp(std::vector<Point> &pts, int n_samples, std::vector<Point> &centers, int n_clusters);

		void random(std::vector<Point> &pts, int n_samples, std::vector<Point> &centers, int n_clusters);
		
	public:
		
		std::vector<int> fit_predict(std::vector< std::vector<float> > &features, int n_clusters, std::string init_method);
};

float KMeans::distance(Point &x, Point &y)
{
	float sum = 0.0f;
	
	for (int i = 0; i < N_FEATURES; ++i)
	{
		sum += (x.data[i] - y.data[i]) * (x.data[i] - y.data[i]);
	}
	
	return sum;
}

int KMeans::nearest(Point &pt, std::vector<Point> &centers, int n_centers, float *d2)
{
	int min_idx = pt.group;
	float min_dist = FLT_MAX;
	
	float dist = 0.0f;
	
	for (int i = 0; i < n_centers; ++i)
	{
		if (min_dist > (dist = distance(pt, centers[i])))
		{
			min_dist = dist;
			min_idx = i;
		}
	}
	
	if (d2) *d2 = min_dist;
	
	return min_idx;
}

void KMeans::kpp(std::vector<Point> &pts, int n_samples, std::vector<Point> &centers, int n_clusters)
{
	centers[0] = pts[ rand() % n_samples ];
	
	float sum;
	float *d = (float *)malloc(n_samples * sizeof(float));
	
	for (int i = 1; i < n_clusters; ++i)
	{
		sum = 0.0f;
		
		for (int j = 0; j < n_samples; ++j)
		{
			nearest(pts[j], centers, i, d+j);
			sum += d[j];
		}
		sum *= rand() / (RAND_MAX - 1.0f);
		
		for (int j = 0; j < n_samples; ++j)
		{
			if ((sum -= d[j]) > 0) continue;
			
			centers[i] = pts[j];
			
			break;
		}
		
	}
	
	for (int j = 0; j < n_samples; ++j)
	{
		pts[j].group = nearest(pts[j], centers, n_clusters, 0);
	}
	
	free(d);
}

void KMeans::random(std::vector<Point> &pts, int n_samples, std::vector<Point> &centers, int n_clusters)
{
	int *Index = (int *)malloc(n_clusters * sizeof(int));
	
	for (int i = 0; i < n_samples; ++i)
	{
		Index[i] = i;
	}
	
	std::random_shuffle(Index, Index+n_clusters);
	
	for (int i = 0; i < n_clusters; ++i)
	{
		centers[i] = pts[Index[i]];
	}
	
	free(Index);
}

std::vector<int> KMeans::fit_predict(std::vector< std::vector<float> > &features, int n_clusters, std::string init_method)
{
	int n_samples = features.size();
	
	if (n_clusters > n_samples)
	{
		printf("number of clusters %d must be less than number of samples %d !", n_clusters, n_samples);
		exit(0);
	}
	
	std::vector<int> labels(n_samples);
	
	std::vector<Point> pts(n_samples);
	
	for (int i = 0; i < n_samples; ++i)
	{
		memcpy(pts[i].data, features[i].data(), N_FEATURES * sizeof(float));
	}
	
	float zeros[N_FEATURES];
	for (int i = 0; i < N_FEATURES; ++i) zeros[i] = 0.0f;
	
	std::vector<Point> cluster_centers(n_clusters);
	
	int iter;
	int changed;
	float inertia;
	float best_inertia = FLT_MAX;
	int stop_msg = n_samples >> 10; //0.1% of n_samples
	
	float* clu_dist = (float*)malloc(sizeof(float) * 1);
	
	for (int n = 0; n < N_INIT; ++n)
	{
		if (init_method == "kmeans++" || init_method == "kmeanspp")
		{
			kpp(pts, n_samples, cluster_centers, n_clusters);
		}
		else if (init_method == "random")
		{
			random(pts, n_samples, cluster_centers, n_clusters);
		}
		else
		{
			printf("Only support kmeans++ and random initial methods! Please choose one of them.");
			exit(0);
		}
		
		iter = 0;
		
		do
		{
			++iter;
			
			for (int i = 0; i < n_clusters; ++i)
			{
				memcpy(cluster_centers[i].data, zeros, N_FEATURES * sizeof(float));
				cluster_centers[i].group = 0;
			}
			
			for (int i = 0; i < n_samples; ++i)
			{
				++cluster_centers[pts[i].group].group;
				
				for (int j = 0; j < N_FEATURES; ++j)
				{
					cluster_centers[pts[i].group].data[j] += pts[i].data[j];
				}
			}
			
			for (int i = 0; i < n_clusters; ++i)
			{
				if (cluster_centers[i].group == 0) continue;
				
				for (int j = 0; j < N_FEATURES; ++j)
				{
					cluster_centers[i].data[j] /= (float)cluster_centers[i].group;
				}
			}
			
			changed = 0;
			
			for (int i = 0; i < n_samples; ++i)
			{
				int min_idx = nearest(pts[i], cluster_centers, n_clusters, clu_dist);
				if (min_idx != pts[i].group)
				{
					++changed;
					pts[i].group = min_idx;
					pts[i].clu_dist = clu_dist[0];
				}
			}
			
		} while (changed > stop_msg && iter <= MAX_ITER);
		
		inertia = 0.0f;
		
		for (int i = 0; i < n_samples; ++i)
		{
			inertia += pts[i].clu_dist;
		}
		
		if (inertia < best_inertia)
		{
			for (int i = 0; i < n_samples; ++i)
			{
				labels[i] = pts[i].group;
			}
			
			best_inertia = inertia;
		}
	}
	
	free(clu_dist);
	
	return labels;
}