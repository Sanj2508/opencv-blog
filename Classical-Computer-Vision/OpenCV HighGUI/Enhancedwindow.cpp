#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class EnhancedWindow {
public:
    int x, y, width, height;
    string title;
    Scalar color;
    Scalar title_color;
    bool minimized;
    bool dragging;
    int offset_x, offset_y;
    int title_bar_height;

    EnhancedWindow(int _x, int _y, int _width, int _height, string _title,
                   Scalar _color = Scalar(60, 60, 60),
                   Scalar _title_color = Scalar(100, 100, 255))
    {
        x = _x;
        y = _y;
        width = _width;
        height = _height;
        title = _title;
        color = _color;
        title_color = _title_color;
        minimized = false;
        dragging = false;
        offset_x = 0;
        offset_y = 0;
        title_bar_height = 35;
    }

    bool inside(int mx, int my) {
        return (mx >= x && mx <= x + width && my >= y && my <= y + height);
    }

    void handleMouse(int event, int mx, int my, int flags) {
        Rect minimize_btn_area(x + width - 30, y, 30, title_bar_height);

        if (event == EVENT_LBUTTONDOWN) {
            if (minimize_btn_area.contains(Point(mx, my))) {
                minimized = !minimized;
            }
            else if (inside(mx, my)) {
                dragging = true;
                offset_x = mx - x;
                offset_y = my - y;
            }
        }
        else if (event == EVENT_LBUTTONUP) {
            dragging = false;
        }
        else if (event == EVENT_MOUSEMOVE && dragging) {
            x = mx - offset_x;
            y = my - offset_y;
        }
    }

    void draw(Mat& frame) {
        int frame_w = frame.cols;
        int frame_h = frame.rows;

        x = max(0, min(x, frame_w - width));
        y = max(0, min(y, frame_h - title_bar_height));

        // Title bar
        rectangle(frame, Point(x, y), Point(x + width, y + title_bar_height), title_color, FILLED);

        // Minimize/Expand button
        rectangle(frame, Point(x + width - 30, y),
                  Point(x + width, y + title_bar_height), Scalar(80, 80, 200), FILLED);

        string btn_symbol = minimized ? "+" : "-";
        putText(frame, btn_symbol, Point(x + width - 20, y + 25),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // Title text
        putText(frame, title, Point(x + 10, y + 25),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // Content area if not minimized
        if (!minimized) {
            rectangle(frame, Point(x, y + title_bar_height),
                      Point(x + width, y + height), color, FILLED);
        }
    }
};

EnhancedWindow panel(100, 100, 400, 300, "Panel", Scalar(50, 50, 70), Scalar(90, 130, 255));
EnhancedWindow options(550, 150, 400, 300, "Options", Scalar(70, 50, 50), Scalar(255, 90, 90));

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    panel.handleMouse(event, x, y, flags);
    options.handleMouse(event, x, y, flags);
}

int main() {
    string window_name = "Enhanced OpenCV UI";
    namedWindow(window_name);
    setMouseCallback(window_name, mouseCallback);

    Mat frame(600, 1000, CV_8UC3, Scalar(35, 35, 35));

    while (true) {
        frame.setTo(Scalar(35, 35, 35));

        panel.draw(frame);
        options.draw(frame);

        imshow(window_name, frame);
        char key = (char)waitKey(20);
        // Press ESC to exit
        if (key == 27) break; 
    }

    destroyAllWindows();
    return 0;
}
