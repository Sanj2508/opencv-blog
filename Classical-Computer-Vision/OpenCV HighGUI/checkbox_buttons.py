import cv2
import numpy as np

# Window Setup
WIDTH, HEIGHT = 1000, 900
img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Global State
selected_language = None
selected_interests = []
message = ""

# Options
languages = ["Python", "C++", "Java"]
interests = ["AI / ML", "Web Dev", "Game Dev", "Data Science"]

# Colors
BG_TOP = (255, 250, 245)
BG_BOTTOM = (230, 230, 240)
BTN_COLOR = (0, 120, 230)
TEXT_COLOR = (30, 30, 30)

# Layout constants
margin_x = 100
margin_y = 120
line_gap = 60
section_gap = 70

# Helper functions
def draw_gradient_background():
    """Draws a smooth vertical gradient background."""
    for y in range(HEIGHT):
        alpha = y / HEIGHT
        color = np.array(BG_TOP) * (1 - alpha) + np.array(BG_BOTTOM) * alpha
        img[y, :] = color

def draw_text(text, pos, size=0.8, color=TEXT_COLOR, bold=False):
    """Draws anti-aliased text."""
    thickness = 2 if bold else 1
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

def draw_radio(x, y, label, selected):
    """Draws a radio button."""
    cv2.circle(img, (x, y), 12, (80, 80, 80), 2)
    if selected:
        cv2.circle(img, (x, y), 6, (0, 0, 180), -1)
    draw_text(label, (x + 30, y + 7), 0.8)

def draw_checkbox(x, y, label, checked):
    """Draws a checkbox."""
    cv2.rectangle(img, (x - 12, y - 12), (x + 12, y + 12), (80, 80, 80), 2)
    if checked:
        cv2.line(img, (x - 8, y), (x - 2, y + 8), (0, 180, 0), 3)
        cv2.line(img, (x - 2, y + 8), (x + 8, y - 10), (0, 180, 0), 3)
    draw_text(label, (x + 30, y + 7), 0.8)

def draw_ui():
    """Draws all UI components and returns button position."""
    draw_gradient_background()
    draw_text("Interactive Learning Survey", (margin_x, 70), 1.2, (0, 0, 0), bold=True)

    # Use of Radio Buttons
    y = margin_y
    draw_text("1. Choose your favorite language:", (margin_x, y), 0.9, bold=True)
    y += line_gap
    for lang in languages:
        draw_radio(margin_x + 20, y, lang, selected_language == lang)
        y += line_gap

    # Checkboxes
    y += section_gap - 20
    draw_text("2. What are you interested in?", (margin_x, y), 0.9, bold=True)
    y += line_gap
    for interest in interests:
        draw_checkbox(margin_x + 20, y, interest, interest in selected_interests)
        y += line_gap

    # Submit button 
    btn_w, btn_h = 200, 60
    btn_x, btn_y = WIDTH // 2 - btn_w // 2, HEIGHT - 180
    cv2.rectangle(img, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), BTN_COLOR, -1)
    draw_text("Submit", (btn_x + 55, btn_y + 38), 0.9, (255, 255, 255), bold=True)

    # Display message
    if message:
        draw_text(message, (margin_x, HEIGHT - 40), 0.8, (0, 100, 0), bold=True)

    return btn_x, btn_y, btn_w, btn_h

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks."""
    global selected_language, selected_interests, message

    if event == cv2.EVENT_LBUTTONDOWN:
        # Radio buttons
        y_pos = margin_y + line_gap
        for lang in languages:
            # vertical bounds for radio circle (24×24 box)
            if (margin_x + 20 - 15) <= x <= (margin_x + 20 + 15) and (y_pos - 15) <= y <= (y_pos + 15):
                selected_language = lang
            y_pos += line_gap

        # Checkboxes
        y_pos = margin_y + (len(languages) + 2) * line_gap + 50
        for interest in interests:
            if (margin_x + 20 - 15) <= x <= (margin_x + 20 + 15) and (y_pos - 15) <= y <= (y_pos + 15):
                if interest in selected_interests:
                    selected_interests.remove(interest)
                else:
                    selected_interests.append(interest)
            y_pos += line_gap

        # Submit Button 
        btn_x, btn_y, btn_w, btn_h = WIDTH // 2 - 100, HEIGHT - 180, 200, 60
        if btn_x <= x <= btn_x + btn_w and btn_y <= y <= btn_y + btn_h:
            if not selected_language:
                message = "Please select a language first!"
            else:
                interests_text = ", ".join(selected_interests) if selected_interests else "None"
                message = f"{selected_language} | Interests: {interests_text}"

# Main Loop 
cv2.namedWindow("Interactive UI", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Interactive UI", mouse_callback)

while True:
    draw_ui()
    cv2.imshow("Interactive UI", img)
     # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cv2.destroyAllWindows()

