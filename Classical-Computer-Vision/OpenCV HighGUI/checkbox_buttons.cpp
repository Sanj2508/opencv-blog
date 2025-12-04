#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>

using namespace cv;
using namespace std;

// Window Setup
const int WIDTH = 1000;
const int HEIGHT = 900;

// Global State
string selected_language;
vector<string> selected_interests;
string message;

// Options
vector<string> languages = {"Python", "C++", "Java"};
vector<string> interests = {"AI / ML", "Web Dev", "Game Dev", "Data Science"};

// Colors
Scalar BG_TOP(255, 250, 245);
Scalar BG_BOTTOM(230, 230, 240);
Scalar BTN_COLOR(0, 120, 230);
Scalar TEXT_COLOR(30, 30, 30);

// Layout
const int margin_x = 100;
const int margin_y = 120;
const int line_gap = 60;
const int section_gap = 70;

// Helper functions
void draw_gradient_background(Mat &img) {
    for (int y = 0; y < HEIGHT; y++) {
        float alpha = float(y) / HEIGHT;
        Vec3b color;
        for (int c = 0; c < 3; c++) {
            color[c] = saturate_cast<uchar>(BG_TOP[c] * (1 - alpha) + BG_BOTTOM[c] * alpha);
        }
        img.row(y).setTo(color);
    }
}

void draw_text(Mat &img, const string& text, Point pos, double size = 0.8, Scalar color = TEXT_COLOR, bool bold = false) {
    int thickness = bold ? 2 : 1;
    putText(img, text, pos, FONT_HERSHEY_SIMPLEX, size, color, thickness, LINE_AA);
}

void draw_radio(Mat &img, int x, int y, const string& label, bool selected) {
    circle(img, Point(x, y), 12, Scalar(80, 80, 80), 2);
    if (selected) circle(img, Point(x, y), 6, Scalar(0, 0, 180), -1);
    draw_text(img, label, Point(x + 30, y + 7), 0.8);
}

void draw_checkbox(Mat &img, int x, int y, const string& label, bool checked) {
    rectangle(img, Point(x - 12, y - 12), Point(x + 12, y + 12), Scalar(80, 80, 80), 2);
    if (checked) {
        line(img, Point(x - 8, y), Point(x - 2, y + 8), Scalar(0, 180, 0), 3);
        line(img, Point(x - 2, y + 8), Point(x + 8, y - 8), Scalar(0, 180, 0), 3);
    }
    draw_text(img, label, Point(x + 30, y + 7), 0.8);
}

tuple<int,int,int,int> draw_ui(Mat &img) {
    draw_gradient_background(img);
    draw_text(img, "Interactive Learning Survey", Point(margin_x, 70), 1.2, Scalar(0,0,0), true);

    int y = margin_y;
    draw_text(img, "1. Choose your favorite language:", Point(margin_x, y), 0.9, Scalar(0,0,0), true);
    y += line_gap;
    for (const auto& lang : languages) {
        draw_radio(img, margin_x + 20, y, lang, selected_language == lang);
        y += line_gap;
    }

    y += section_gap - 20;
    draw_text(img, "2. What are you interested in?", Point(margin_x, y), 0.9, Scalar(0,0,0), true);
    y += line_gap;
    for (const auto& interest : interests) {
        draw_checkbox(img, margin_x + 20, y, interest, find(selected_interests.begin(), selected_interests.end(), interest) != selected_interests.end());
        y += line_gap;
    }

    int btn_w = 200, btn_h = 60;
    int btn_x = WIDTH/2 - btn_w/2, btn_y = HEIGHT - 180;
    rectangle(img, Point(btn_x, btn_y), Point(btn_x + btn_w, btn_y + btn_h), BTN_COLOR, -1);
    draw_text(img, "Submit", Point(btn_x + 55, btn_y + 38), 0.9, Scalar(255,255,255), true);

    if (!message.empty()) draw_text(img, message, Point(margin_x, HEIGHT - 40), 0.8, Scalar(0,100,0), true);

    return make_tuple(btn_x, btn_y, btn_w, btn_h);
}

void mouse_callback(int event, int x, int y, int flags, void* param) {
    if (event != EVENT_LBUTTONDOWN) return;

    // Retrieve button coordinates
    int btn_x = WIDTH/2 - 100, btn_y = HEIGHT - 180, btn_w = 200, btn_h = 60;

    // Radio buttons
    int y_pos = margin_y + line_gap;
    for (auto &lang : languages) {
        if (x >= margin_x + 20 - 15 && x <= margin_x + 20 + 15 && y >= y_pos - 15 && y <= y_pos + 15)
            selected_language = lang;
        y_pos += line_gap;
    }

    // Checkboxes
    y_pos = margin_y + (languages.size() + 2)*line_gap + 50;
    for (auto &interest : interests) {
        if (x >= margin_x + 20 - 15 && x <= margin_x + 20 + 15 && y >= y_pos - 15 && y <= y_pos + 15) {
            auto it = find(selected_interests.begin(), selected_interests.end(), interest);
            if (it != selected_interests.end()) selected_interests.erase(it);
            else selected_interests.push_back(interest);
        }
        y_pos += line_gap;
    }

    // Submit button
    if (x >= btn_x && x <= btn_x + btn_w && y >= btn_y && y <= btn_y + btn_h) {
        if (selected_language.empty()) message = "Please select a language first!";
        else {
            string interests_text = selected_interests.empty() ? "None" : selected_interests[0];
            for (size_t i=1; i<selected_interests.size(); i++) interests_text += ", " + selected_interests[i];
            message = selected_language + " | Interests: " + interests_text;
        }
    }
}

int main() {
    namedWindow("Interactive UI", WINDOW_AUTOSIZE);
    setMouseCallback("Interactive UI", mouse_callback);

    Mat img;
    while (true) {
        // reset canvas each frame
        img = Mat::ones(HEIGHT, WIDTH, CV_8UC3); 
        draw_ui(img);
        imshow("Interactive UI", img);
        // ESC to exit
        if (waitKey(30) == 27) break; 
    }

    destroyAllWindows();
    return 0;
}
