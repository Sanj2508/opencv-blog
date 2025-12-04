#include <opencv2/opencv.hpp>
#include <iostream>
#include <stack>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;

// Declare Global variables
Mat canvas;
bool drawing = false;
string mode = "rectangle";
Point startPoint(-1, -1);
stack<Mat> undoStack, redoStack;

// Trackbar values
int r_val = 0, g_val = 0, b_val = 0;
int thickness = 2;
int fillMode = 0; 

// Dummy callback for trackbars
void nothing(int, void*) {}

// Mouse callback function
void drawShape(int event, int x, int y, int flags, void* userdata) {
    Scalar color(b_val, g_val, r_val);
    int actualThickness = (fillMode && (mode == "rectangle" || mode == "circle")) ? FILLED : thickness;

    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        startPoint = Point(x, y);
        undoStack.push(canvas.clone());
    }
    else if (event == EVENT_MOUSEMOVE && drawing) {
        Mat temp = canvas.clone();
        if (mode == "rectangle")
            rectangle(temp, startPoint, Point(x, y), color, actualThickness);
        else if (mode == "circle") {
            int radius = (int)sqrt(pow(x - startPoint.x, 2) + pow(y - startPoint.y, 2));
            circle(temp, startPoint, radius, color, actualThickness);
        }
        else if (mode == "line")
            line(temp, startPoint, Point(x, y), color, thickness);
        else if (mode == "eraser")
            line(temp, startPoint, Point(x, y), Scalar(255, 255, 255), thickness);

        imshow("Drawing Canvas", temp);
    }
    else if (event == EVENT_LBUTTONUP) {
        drawing = false;
        if (mode == "rectangle")
            rectangle(canvas, startPoint, Point(x, y), color, actualThickness);
        else if (mode == "circle") {
            int radius = (int)sqrt(pow(x - startPoint.x, 2) + pow(y - startPoint.y, 2));
            circle(canvas, startPoint, radius, color, actualThickness);
        }
        else if (mode == "line")
            line(canvas, startPoint, Point(x, y), color, thickness);
        else if (mode == "eraser")
            line(canvas, startPoint, Point(x, y), Scalar(255, 255, 255), thickness);

        while (!redoStack.empty()) redoStack.pop();
    }
}

int main() {
    string imagePath = "C:/Users/samru/PycharmProjects/OpenCV HighGUI/image.jpg";
    if (filesystem::exists(imagePath)) {
        canvas = imread(imagePath);
        resize(canvas, canvas, Size(800, 600));
    } else {
        canvas = Mat(700, 1000, CV_8UC3, Scalar(255, 255, 255));
    }

    namedWindow("Drawing Canvas");

    // Trackbars 
    createTrackbar("R", "Drawing Canvas", &r_val, 255, nothing);
    createTrackbar("G", "Drawing Canvas", &g_val, 255, nothing);
    createTrackbar("B", "Drawing Canvas", &b_val, 255, nothing);
    createTrackbar("Thickness", "Drawing Canvas", &thickness, 30, nothing);
    createTrackbar("Fill", "Drawing Canvas", &fillMode, 1, nothing); // use fillMode

    setMouseCallback("Drawing Canvas", drawShape);

    cout << R"(
   OpenCV Drawing GUI (C++)
   Left-click and drag to draw
   Trackbars: R, G, B, Thickness, Fill
   Keyboard Shortcuts:
   r - Rectangle
   c - Circle
   l - Line
   e - Eraser
   z - Undo
   y - Redo
   x - Clear
   s - Save
   ESC - Exit
)" << endl;

    // Main loop
    while (true) {
        Mat display = canvas.clone();
        string fillStatus = (fillMode == 1) ? "ON" : "OFF";
        putText(display, "Mode: " + mode + " | Fill: " + fillStatus, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 50, 50), 2);

        imshow("Drawing Canvas", display);
        int key = waitKey(1) & 0xFF;

        if (key == 27) break; // ESC
        else if (key == 'r') mode = "rectangle";
        else if (key == 'c') mode = "circle";
        else if (key == 'l') mode = "line";
        else if (key == 'e') mode = "eraser";
        else if (key == 'x') {
            undoStack.push(canvas.clone());
            canvas = Scalar(255, 255, 255);
            cout << "Canvas cleared" << endl;
        }
        else if (key == 'z' && !undoStack.empty()) {
            redoStack.push(canvas.clone());
            canvas = undoStack.top();
            undoStack.pop();
            cout << "Undo" << endl;
        }
        else if (key == 'y' && !redoStack.empty()) {
            undoStack.push(canvas.clone());
            canvas = redoStack.top();
            redoStack.pop();
            cout << "Redo" << endl;
        }
        else if (key == 's') {
            string savePath = (filesystem::current_path() / "drawing_saved.png").string();
            imwrite(savePath, canvas);
            cout << "Saved: " << savePath << endl;
        }
    }

    string autoSave = (filesystem::current_path() / "drawing_autosave.png").string();
    imwrite(autoSave, canvas);
    cout << "Auto-saved: " << autoSave << endl;

    destroyAllWindows();
    return 0;
}
