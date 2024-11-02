#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>
#include <thread>
#include <fstream>  // For file operations


constexpr int kClustersNumber = 5;
constexpr int kDataSize = 1000 ;
constexpr int kIterNo = 50;

std::vector<std::pair<int,int>> points;
std::vector<std::pair<int,int>> centroids;

std::vector<std::vector<std::pair<int,int>>> clusters = std::vector<std::vector<std::pair<int,int>>>(kClustersNumber);

std::mutex vector_mutex;

void InitializeCentroids(){

    int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0){
        num_threads = 1;
    }

    int chunk_size = kClustersNumber / num_threads;
    int reminder_chunk = kClustersNumber % num_threads;

    auto worker = [](int begin, int end) {
        auto thread_id = std::this_thread::get_id();
        std::hash<std::thread::id> hasher;
        size_t hashed_id = hasher(thread_id);

        std::srand(hashed_id^std::time(0));

        for (int i = begin; i < end; i++) {
            std::pair<int, int> centroid = std::pair<int, int>(std::rand(), std::rand());
            std::lock_guard<std::mutex> lock(vector_mutex);
            centroids.push_back(centroid);
        }
    };

    std::vector<std::thread> threads;
    int current = 0;
    for(int i=0;i<num_threads;i++){
        int begin = current;
        int end = current + chunk_size;
        if(i < reminder_chunk){
            end+=1;
        }
        threads.emplace_back(worker,begin,end);
        current = end;
    }

    for (std::thread &thread: threads) {
        thread.join();
    }

}


void InitializePoints(){

    int num_threads = std::thread::hardware_concurrency();
    if(num_threads==0)
        num_threads = 1;

    int chunk_size = kDataSize/num_threads;
    int chunk_size_reminder = kDataSize%num_threads;

    auto worker = [](int begin, int end) {
        auto thread_id = std::this_thread::get_id();
        std::hash<std::thread::id> hasher;
        size_t hashed_id = hasher(thread_id);

        std::srand(hashed_id^std::time(0));


        for (int i = begin; i < end; i++) {
            std::pair<int, int> point = std::pair<int, int>(std::rand(), std::rand());
            std::lock_guard<std::mutex> lock(vector_mutex);
            points.push_back(point);
        }
    };

    std::vector<std::thread> threads;
    int current = 0;
    for(int i=0;i<num_threads;i++){
        int begin = current;
        int end = current + chunk_size;

        if(i<chunk_size_reminder)
            end+=1;

        threads.emplace_back(worker,begin,end);
        current = end;
    }

    for (auto &thread: threads) {
        thread.join();
    }

}


int EuclideanSquared(std::pair<int, int> x, std::pair<int,int> y){
    return (x.first-y.first)*(x.first-y.first) + (x.second-y.second)*(x.second-y.second);
}

std::vector<int> CalculateDistancesPointCentroids(std::pair<int,int> point){
    std::vector<int> distances;
    for (const auto &centroid: centroids) {
        int distance = EuclideanSquared(point, centroid);
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

    int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0)
        num_threads = 2;

    int chunk_size = kDataSize/num_threads;
    int chunk_size_reminder = kDataSize%num_threads;



    auto worker = [](int begin, int end) {
        for (int point_no = begin; point_no < end; point_no++) {
            std::vector<int> distance = CalculateDistancesPointCentroids(points[point_no]);
            auto min_it = std::min_element(distance.begin(), distance.end());
            int min_index = std::distance(distance.begin(), min_it);

            std::lock_guard<std::mutex> lock(vector_mutex);
            clusters[min_index].push_back(points[point_no]);
        }
    };

    int current = 0;
    std::vector<std::thread> threads;
    for(int i=0;i<num_threads;i++){
        int begin = current;
        int end = current+chunk_size;

        if(i<chunk_size_reminder){
            end+=1;
        }

        threads.emplace_back(worker,begin,end);
        current = end;
    }

    for (auto &thread: threads) {
        thread.join();
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
