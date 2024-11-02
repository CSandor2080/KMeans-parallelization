#include <algorithm>
#include <ctime>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <thread>

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

        std::srand(hashed_id^std::time(nullptr));

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


std::vector<int> CalculateDistancesPointCentroids(std::pair<int,int> point){
    std::vector<int> distances;

    __m256i vec_centroids_first = _mm256_set_epi32(
            0, 0, 0,
            centroids[4].first,
            centroids[3].first,
            centroids[2].first,
            centroids[1].first,
            centroids[0].first
    );

    __m256i vec_centroids_second = _mm256_set_epi32(
            0, 0, 0,
            centroids[4].second,
            centroids[3].second,
            centroids[2].second,
            centroids[1].second,
            centroids[0].second

    );
    //(point.first - centroid.first) * (point.first - centroid.first)
    vec_centroids_first = _mm256_sub_epi32(vec_centroids_first, _mm256_set1_epi32(point.first));
    vec_centroids_first = _mm256_mullo_epi32(vec_centroids_first, vec_centroids_first);

    //(point.second - centroid.second) * (point.second - centroid.second)
    vec_centroids_second = _mm256_sub_epi32(vec_centroids_second, _mm256_set1_epi32(point.second));
    vec_centroids_second = _mm256_mullo_epi32(vec_centroids_second, vec_centroids_second);

    __m256i vec_point_to_centroids_distance;
    vec_point_to_centroids_distance = _mm256_add_epi32(vec_centroids_first, vec_centroids_second);

    int temp[8];

    _mm256_storeu_si256((__m256i*)temp,vec_point_to_centroids_distance);
    for(int j=0;j<5;j++)
        distances.push_back(temp[j]);


    return distances;

}

std::pair<int, int> CalculateMean(std::vector<std::pair<int, int>> &pts) {
    __m256i vec_first_rez = _mm256_setzero_si256();
    __m256i vec_second_rez = _mm256_setzero_si256();
    int i = 0;
    const int kDataSize = pts.size();

    for (; i + 7 < kDataSize; i += 8) {
        __m256i vec_first = _mm256_set_epi32(
                pts[i].first, pts[i + 1].first, pts[i + 2].first, pts[i + 3].first,
                pts[i + 4].first, pts[i + 5].first, pts[i + 6].first, pts[i + 7].first
        );
        __m256i vec_second = _mm256_set_epi32(
                pts[i].second, pts[i + 1].second, pts[i + 2].second, pts[i + 3].second,
                pts[i + 4].second, pts[i + 5].second, pts[i + 6].second, pts[i + 7].second
        );

        vec_first_rez = _mm256_add_epi32(vec_first_rez, vec_first);
        vec_second_rez = _mm256_add_epi32(vec_second_rez, vec_second);
    }

    alignas(32) int first_rez[8] = {0};
    alignas(32) int second_rez[8] = {0};

    _mm256_storeu_si256((__m256i*)(first_rez), vec_first_rez);
    _mm256_storeu_si256((__m256i*)(second_rez), vec_second_rez);

    for (; i < kDataSize; i++) {
        first_rez[0] += pts[i].first;
        second_rez[0] += pts[i].second;
    }

    int first_elements_sum = std::accumulate(std::begin(first_rez), std::end(first_rez), 0);
    int second_elements_sum = std::accumulate(std::begin(second_rez), std::end(second_rez), 0);


    std::pair<int, int> mean;
    mean.first = first_elements_sum / kDataSize;
    mean.second = second_elements_sum / kDataSize;

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
