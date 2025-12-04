import cv2
import numpy as np
import os


IMAGE_PATH = r"C:\Users\samru\PycharmProjects\OpenCV HighGUI\image.jpg"

# Load image or create blank canvas
if os.path.exists(IMAGE_PATH):
    canvas = cv2.imread(IMAGE_PATH)
    canvas = cv2.resize(canvas, (800, 600))
else:
    canvas = np.ones((700, 1000, 3), dtype=np.uint8) * 255

drawing = False
# Default mode
mode = 'rectangle'  
start_x, start_y = -1, -1
undo_stack = []
redo_stack = []

# Trackbar callback 
def nothing(x):
    pass

cv2.namedWindow('Drawing Canvas')

# Trackbars
cv2.createTrackbar('R', 'Drawing Canvas', 0, 255, nothing)
cv2.createTrackbar('G', 'Drawing Canvas', 0, 255, nothing)
cv2.createTrackbar('B', 'Drawing Canvas', 0, 255, nothing)
cv2.createTrackbar('Thickness', 'Drawing Canvas', 2, 30, nothing)
cv2.createTrackbar('Fill', 'Drawing Canvas', 0, 1, nothing)  

# Mouse callback function 
def draw_shape(event, x, y, flags, param):
    global start_x, start_y, drawing, mode, canvas, undo_stack

    r = cv2.getTrackbarPos('R', 'Drawing Canvas')
    g = cv2.getTrackbarPos('G', 'Drawing Canvas')
    b = cv2.getTrackbarPos('B', 'Drawing Canvas')
    thickness = cv2.getTrackbarPos('Thickness', 'Drawing Canvas')
    fill = cv2.getTrackbarPos('Fill', 'Drawing Canvas') == 1  

    color = (b, g, r)  
    actual_thickness = -1 if fill and mode in ['rectangle', 'circle'] else thickness

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        # Save current state
        undo_stack.append(canvas.copy())  

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_canvas = canvas.copy()
        if mode == 'rectangle':
            cv2.rectangle(temp_canvas, (start_x, start_y), (x, y), color, actual_thickness)
        elif mode == 'circle':
            radius = int(((x - start_x)**2 + (y - start_y)**2)**0.5)
            cv2.circle(temp_canvas, (start_x, start_y), radius, color, actual_thickness)
        elif mode == 'line':
            cv2.line(temp_canvas, (start_x, start_y), (x, y), color, thickness)
        elif mode == 'eraser':
            cv2.line(temp_canvas, (start_x, start_y), (x, y), (255, 255, 255), thickness)
        cv2.imshow('Drawing Canvas', temp_canvas)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 'rectangle':
            cv2.rectangle(canvas, (start_x, start_y), (x, y), color, actual_thickness)
        elif mode == 'circle':
            radius = int(((x - start_x)**2 + (y - start_y)**2)**0.5)
            cv2.circle(canvas, (start_x, start_y), radius, color, actual_thickness)
        elif mode == 'line':
            cv2.line(canvas, (start_x, start_y), (x, y), color, thickness)
        elif mode == 'eraser':
            cv2.line(canvas, (start_x, start_y), (x, y), (255, 255, 255), thickness)

        redo_stack.clear()

cv2.setMouseCallback('Drawing Canvas', draw_shape)

print("""
   OpenCV Drawing GUI
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
""")

# Main loop 
while True:
    display_canvas = canvas.copy()
    fill_status = "ON" if cv2.getTrackbarPos('Fill', 'Drawing Canvas') == 1 else "OFF"
    cv2.putText(display_canvas, f"Mode: {mode.upper()} | Fill: {fill_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    cv2.imshow('Drawing Canvas', display_canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('r'):
        mode = 'rectangle'
    elif key == ord('c'):
        mode = 'circle'
    elif key == ord('l'):
        mode = 'line'
    elif key == ord('e'):
        mode = 'eraser'
    elif key == ord('x'):
        undo_stack.append(canvas.copy())
        canvas[:] = 255
        print("Canvas cleared")
    elif key == ord('z') and undo_stack:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()
        print("Undo")
    elif key == ord('y') and redo_stack:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()
        print("Redo")
    elif key == ord('s'):
        save_path = os.path.join(os.getcwd(), "drawing_saved.png")
        cv2.imwrite(save_path, canvas)
        print(f" Saved: {save_path}")

# Auto-save on exit
auto_save_path = os.path.join(os.getcwd(), "drawing_autosave.png")
cv2.imwrite(auto_save_path, canvas)
print(f"Auto-saved: {auto_save_path}")

cv2.destroyAllWindows()
