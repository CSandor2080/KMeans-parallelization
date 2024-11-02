import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors


# centroids_file = 'serial/centroids.txt'
# points_file = 'serial/points.txt'

centroids_file = 'cmake-build-debug/centroids.txt'
points_file = 'cmake-build-debug/points.txt'



def read_centroids(file_path):
    centroids = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == "" or line.startswith("Centroids:"):
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            x, y = map(int, parts)
            centroids.append((x, y))
    return centroids

def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == "" or line.startswith("Dataset:"):
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            x, y = map(int, parts)
            points.append((x, y))
    return points


def assign_clusters(centroids, points):
    clusters = defaultdict(list)
    for point in points:
        distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(point)
    return clusters





def plot_clusters_with_hulls(clusters, centroids):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    if len(clusters) > len(colors):

        colors = colors * (len(clusters) // len(colors) + 1)

    plt.figure(figsize=(12, 10))


    for cluster_idx, points in clusters.items():
        cluster_points = np.array(points)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=50, c=mcolors.TABLEAU_COLORS[colors[cluster_idx]],
                    label=f'Cluster {cluster_idx+1}', alpha=0.6, edgecolors='w')


        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]

            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            plt.plot(hull_points[:, 0], hull_points[:, 1],
                     c=mcolors.TABLEAU_COLORS[colors[cluster_idx]], linestyle='--', linewidth=2)
        elif len(cluster_points) == 2:

            plt.plot(cluster_points[:, 0], cluster_points[:, 1],
                     c=mcolors.TABLEAU_COLORS[colors[cluster_idx]], linestyle='--', linewidth=2)



    centroids_np = np.array(centroids)
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1],
                s=300, c='black', marker='X', label='Centroids', edgecolors='yellow')


    for idx, (x, y) in enumerate(centroids):
        plt.text(x, y, f'  C{idx+1}', fontsize=12, fontweight='bold', color='black')

    plt.title('K-means Clustering Results with Convex Hulls', fontsize=16)
    plt.xlabel('X-coordinate', fontsize=14)
    plt.ylabel('Y-coordinate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():


    centroids = read_centroids(centroids_file)
    points = read_points(points_file)

    if not centroids:
        print("No centroids found. Please check centroids.txt.")
        return
    if not points:
        print("No points found. Please check points.txt.")
        return


    clusters = assign_clusters(centroids, points)


    plot_clusters_with_hulls(clusters, centroids)

if __name__ == "__main__":
    main()
