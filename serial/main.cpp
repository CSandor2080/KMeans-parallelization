#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

constexpr int kClustersNumber = 5;
constexpr int kDataSize = 100;
constexpr int kIterNo = 10;

std::vector<std::pair<int,int>> points;
std::vector<std::pair<int,int>> centroids;

std::vector<std::vector<std::pair<int,int>>> clusters = std::vector<std::vector<std::pair<int,int>>>(kClustersNumber);
std::vector<int> cent = std::vector<int>(kClustersNumber);

void InitializeCentroids(){
    std::srand(std::time(0));

    for(int i=0;i<kClustersNumber;i++){
        std::pair<int,int> centroid = std::pair<int,int>(std::rand(),std::rand());
        centroids.push_back(centroid);
    }
}


void InitializePoints(){
    std::srand(std::time(0));

    for(int i=0;i<kDataSize;i++){
        std::pair<int,int> point = std::pair<int,int>(std::rand(),std::rand());
        points.push_back(point);
    }
}

//Euclidean squared distancea
int EucliadeanSquared(std::pair<int, int> x, std::pair<int,int> y){
    return (x.first-y.first)*(x.first-y.first) + (x.second-y.second)*(x.second-y.second);
}

std::vector<int> CalculateDistancesPointCentroids(std::pair<int,int> point){
    std::vector<int> distances;
    for (const auto &centroid: centroids) {
        int distance = EucliadeanSquared(point, centroid);
        distances.push_back(distance);
    }

    return distances;

}

std::pair<int,int> CalculateMean(std::vector<std::pair<int, int>> &pts){
    std::pair<int, int> mean = std::pair<int,int>(0,0);
    for (const auto &pt: pts) {
        mean.first += pt.first;
        mean.second += pt.second;
    }

    mean.first/=pts.size();
    mean.second/=pts.size();

    return mean;
}



void Kmeans(){
    for(int point_no=0;point_no<kDataSize;point_no++){
        std::vector<int> distance = CalculateDistancesPointCentroids(points[point_no]);
        auto min_it = std::min_element(distance.begin(), distance.end());
        int min_index = std::distance(distance.begin(), min_it);

        clusters[min_index].push_back(points[point_no]);
    }

    for(int cluster_no=0;cluster_no<kClustersNumber;cluster_no++){
        if(!clusters[cluster_no].empty()){
            centroids[cluster_no] = CalculateMean(clusters[cluster_no]);
        }
    }
}

void ShowCentroids(){
    std::ofstream centroidFile("centroids.txt");
    if (!centroidFile.is_open()) {
        std::cerr << "Failed to open centroids.txt for writing." << std::endl;
        return;
    }

    centroidFile << "Centroids:" << std::endl;
    for (const auto &item: centroids) {
        centroidFile << item.first << " " << item.second << std::endl;
    }

    centroidFile.close();
    std::cout << "Centroids have been written to centroids.txt" << std::endl;
}

// Modified ShowPoints function
void ShowPoints(){
    std::ofstream pointsFile("points.txt");
    if (!pointsFile.is_open()) {
        std::cerr << "Failed to open points.txt for writing." << std::endl;
        return;
    }

    pointsFile << "Dataset:" << std::endl;
    for (const auto &item: points) {
        pointsFile << item.first << " " << item.second << std::endl;
    }

    pointsFile.close();
    std::cout << "Dataset points have been written to points.txt" << std::endl;
}


int main() {
    InitializeCentroids();
    InitializePoints();
    for(int i=0;i<kIterNo;i++) {
        Kmeans();
    }
   ShowCentroids();
    ShowPoints();
}
