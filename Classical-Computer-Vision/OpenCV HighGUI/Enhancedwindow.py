import cv2
import numpy as np

class EnhancedWindow:
    def __init__(self, x, y, width, height, title, color=(60, 60, 60), title_color=(100, 100, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.color = color
        self.title_color = title_color
        self.minimized = False
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self.title_bar_height = 35

    def inside(self, mx, my):
        """Check if mouse is inside the full window area"""
        return self.x <= mx <= self.x + self.width and self.y <= my <= self.y + self.height

    def handle_mouse(self, event, mx, my, flags):
        """Handle drag and minimize button"""
        # Minimize button area (top-right corner)
        minimize_btn_area = (self.x + self.width - 30, self.y, self.x + self.width, self.y + self.title_bar_height)

        if event == cv2.EVENT_LBUTTONDOWN:
            if minimize_btn_area[0] <= mx <= minimize_btn_area[2] and minimize_btn_area[1] <= my <= minimize_btn_area[3]:
                self.minimized = not self.minimized
            elif self.inside(mx, my):
                self.dragging = True
                self.offset_x = mx - self.x
                self.offset_y = my - self.y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.x = mx - self.offset_x
            self.y = my - self.offset_y

    def draw(self, frame):
        """Draw the window"""
        h, w = frame.shape[:2]
        self.x = np.clip(self.x, 0, w - self.width)
        self.y = np.clip(self.y, 0, h - self.title_bar_height)

        # Draw title bar
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.title_bar_height), self.title_color, -1)

        # Draw minimize/expand button
        cv2.rectangle(frame, (self.x + self.width - 30, self.y),
                      (self.x + self.width, self.y + self.title_bar_height), (80, 80, 200), -1)
        # Draw horizontal line instead of "-" character
        btn_symbol = '+' if self.minimized else '-'
        cv2.putText(frame, btn_symbol, (self.x + self.width - 20, self.y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw title text (optional, can be removed)
        cv2.putText(frame, self.title, (self.x + 10, self.y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw content area if not minimized
        if not self.minimized:
            cv2.rectangle(frame, (self.x, self.y + self.title_bar_height),
                          (self.x + self.width, self.y + self.height), self.color, -1)


def main():
    window_name = "Enhanced OpenCV UI"
    cv2.namedWindow(window_name)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    frame[:] = (35, 35, 35)

    panel = EnhancedWindow(100, 100, 400, 300, "Panel", color=(50, 50, 70), title_color=(90, 130, 255))
    options = EnhancedWindow(550, 150, 400, 300, "Options", color=(70, 50, 50), title_color=(255, 90, 90))

    def mouse_callback(event, x, y, flags, param):
        panel.handle_mouse(event, x, y, flags)
        options.handle_mouse(event, x, y, flags)

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        frame[:] = (35, 35, 35)
        panel.draw(frame)
        options.draw(frame)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
