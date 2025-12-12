#define CERES_FOUND 1

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;


// Read image list
static bool loadImageList(const string &filename, vector<String> &images) {
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cerr << "Unable to open image list file: " << filename << endl;
        return false;
    }
    string line;
    while (getline(file, line)) {
        if (!line.empty()) images.push_back(line);
    }
    return !images.empty();
}

// Save points + camera centers + camera baseline to PLY
// - Points are green, Camera centers are red, Baseline edges between cameras

static bool saveSceneAsPLY(const string &filename,
                           const vector<Vec3f> &points,
                           const vector<Vec3f> &camCenters)
{
    ofstream ofs(filename.c_str());
    if (!ofs.is_open()) {
        cerr << "[ERROR] Could not open " << filename << " for writing." << endl;
        return false;
    }

    const size_t numPoints = points.size();
    const size_t numCams   = camCenters.size();
    const size_t numVerts  = numPoints + numCams;
    const size_t numEdges  = (numCams > 1) ? (numCams - 1) : 0;

    // Header
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << numVerts << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    if (numEdges > 0) {
        ofs << "element edge " << numEdges << "\n";
        ofs << "property int vertex1\n";
        ofs << "property int vertex2\n";
    }
    ofs << "end_header\n";

    // Vertices: first all scene points (green)
    for (const auto &p : points) {
        ofs << p[0] << " " << p[1] << " " << p[2] << " "
            << 0   << " " << 255 << " " << 0   << "\n";  // green
    }

    // Then camera centers (red)
    for (const auto &c : camCenters) {
        ofs << c[0] << " " << c[1] << " " << c[2] << " "
            << 255 << " " << 0   << " " << 0   << "\n";  // red
    }

    // Edges: baseline connecting camera centers in order
    // Note: camera vertices start at index numPoints
    if (numEdges > 0) {
        for (size_t i = 0; i + 1 < numCams; ++i) {
            int v1 = static_cast<int>(numPoints + i);
            int v2 = static_cast<int>(numPoints + i + 1);
            ofs << v1 << " " << v2 << "\n";
        }
    }

    ofs.close();
    cout << "[INFO] Saved scene (points + cameras + baseline) to " << filename << endl;
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        cout << "Usage: " << argv[0]
             << " image_paths.txt f cx cy\n"
             << "Example (Temple): " << argv[0]
             << " image_paths.txt 800 400 225\n";
        return 0;
    }

    string listFile = argv[1];
    double f  = atof(argv[2]);
    double cx = atof(argv[3]);
    double cy = atof(argv[4]);

    vector<String> imagePaths;
    if (!loadImageList(listFile, imagePaths)) return -1;
    cout << "[INFO] Loaded " << imagePaths.size() << " images\n";

    Mat first = imread(imagePaths[0], IMREAD_COLOR);
    if (first.empty()) {
        cerr << "Failed to read first image\n";
        return -1;
    }
    cout << "[INFO] First image size: " << first.cols << " x " << first.rows << endl;

    Matx33d K(f, 0, cx,
              0, f, cy,
              0, 0, 1);
    cout << "[INFO] K:\n" << Mat(K) << endl;

    bool isProjective = true;
    vector<Mat> Rs, ts, points3d;
    cout << "[INFO] Running SFM reconstruction...\n";
    cv::sfm::reconstruct(imagePaths, Rs, ts, K, points3d, isProjective);
    cout << "[INFO] Reconstruction complete. Cameras: " << Rs.size()
         << ", points: " << points3d.size() << endl;

    // Build point cloud for Viz / PLY 
    vector<Vec3f> cloud;
    cloud.reserve(points3d.size());
    for (const Mat &p : points3d) {
        if (p.total() == 3 && p.rows * p.cols == 3 && checkRange(p)) {
            cloud.emplace_back((float)p.at<double>(0),
                               (float)p.at<double>(1),
                               (float)p.at<double>(2));
        }
    }

    // Build camera trajectory for Viz 
    vector<Affine3d> path;
    path.reserve(Rs.size());
    for (size_t i = 0; i < Rs.size(); ++i)
        path.emplace_back(Rs[i], ts[i]);

    // Compute camera centers for PLY (world coordinates)
    // If X_cam = R * X_world + t, then camera center in world is C = -R^T * t
    vector<Vec3f> camCenters;
    camCenters.reserve(Rs.size());
    for (size_t i = 0; i < Rs.size(); ++i) {
        Mat R = Rs[i];   // 3x3, CV_64F
        Mat t = ts[i];   // 3x1, CV_64F

        Mat Rt = R.t();  // transpose
        Mat C  = -Rt * t;

        float X = static_cast<float>(C.at<double>(0));
        float Y = static_cast<float>(C.at<double>(1));
        float Z = static_cast<float>(C.at<double>(2));
        camCenters.emplace_back(X, Y, Z);
    }

    // Save everything to PLY for MeshLab 
    if (!cloud.empty()) {
        saveSceneAsPLY("sfm.ply", cloud, camCenters);
    } else {
        cout << "[WARN] Point cloud is empty, not saving PLY." << endl;
    }

    // Viz visualization
    viz::Viz3d window("Sparse Scene Reconstruction");
    window.setBackgroundColor();
    window.showWidget("coordinate", viz::WCoordinateSystem(0.5));

    if (!cloud.empty()) {
        viz::WCloud cloudW(cloud, viz::Color::green());
        cloudW.setRenderingProperty(viz::POINT_SIZE, 2.0);
        window.showWidget("cloud", cloudW);
    }

    if (!path.empty()) {
        window.showWidget("traj",
            viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::yellow()));
        window.showWidget("frustums",
            viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::white()));
        window.setViewerPose(path[0]);
    }

    window.spin();
    return 0;
}
