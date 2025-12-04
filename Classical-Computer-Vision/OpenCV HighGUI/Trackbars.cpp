#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Dummy callback function for the trackbar
void nothing(int, void*) {}

int main() {
    string image_path = "wild_animals.jpg";

    // Check if the image exists and load it
    Mat img = imread(image_path);
    if (img.empty()) {
        cerr << "Error: Image not found or unable to open at " << image_path << endl;
        return -1;
    }

    // Create a named window
    namedWindow("Trackbar Window", WINDOW_AUTOSIZE);

    // Create a trackbar
    int threshold_value = 0;
    createTrackbar("Threshold", "Trackbar Window", &threshold_value, 255, nothing);

    while (true) {
        // Apply threshold to the image
        Mat thresh_img;
        threshold(img, thresh_img, threshold_value, 255, THRESH_BINARY);

        // Show the result
        imshow("Trackbar Window", thresh_img);

        // Break loop on 'Esc' key
        int key = waitKey(1);
        if (key == 27) { 
            break;
        }
    }

    destroyAllWindows();
    return 0;
}
