import cv2
import numpy as np

A_SHIFT = 0
B_SHIFT = 0
OMEGA = 0.75
T_MIN = 0.35
CLAHE_CLIP = 1.2
RED_STRENGTH = 30

# trackbar callback
def on_trackbar(val):
    global A_SHIFT, B_SHIFT, OMEGA, CLAHE_CLIP, RED_STRENGTH
    try:
        A_SHIFT = cv2.getTrackbarPos("A Shift", "Enhanced") - 50
        B_SHIFT = cv2.getTrackbarPos("B Shift", "Enhanced") - 50
        OMEGA = cv2.getTrackbarPos("Omega", "Enhanced") / 100.0
        CLAHE_CLIP = cv2.getTrackbarPos("CLAHE Clip", "Enhanced") / 10.0
        RED_STRENGTH = cv2.getTrackbarPos("Red Boost", "Enhanced")
    except cv2.error:
        return


# Enhancement functions
def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    A = A - (np.mean(A) - 128) + A_SHIFT
    B = B - (np.mean(B) - 128) + B_SHIFT

    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def restore_red(img):
    b, g, r = cv2.split(img)
    boost = cv2.equalizeHist(r)
    strength = RED_STRENGTH / 100.0
    r_new = cv2.addWeighted(r, 1 - strength, boost, strength, 0)
    return cv2.merge([b, g, r_new])


def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(CLAHE_CLIP, 0.1))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


def dehaze(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    A = np.percentile(gray, 95)
    t = 1 - OMEGA * (gray / A)
    t = np.clip(t, T_MIN, 1.0)
    t = cv2.merge([t, t, t])
    J = (img.astype(np.float32) - A) / t + A
    return np.clip(J, 0, 255).astype(np.uint8)


def sharpen(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.addWeighted(img, 1.2, blur, -0.2, 0)


def gamma_correct(img, g=1.1):
    inv = 1.0 / g
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def enhance_underwater(img):
    img = white_balance(img)
    img = restore_red(img)
    img = clahe_enhance(img)
    img = dehaze(img)
    img = sharpen(img)
    img = gamma_correct(img)
    return img


# SAVE FULL VIDEO 
def save_enhanced_video(video_path):
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("enhanced_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced = enhance_underwater(frame)
        out.write(enhanced)

    out.release()
    cap.release()


# Main application loop
def run_app(image_path, video_path):

    # cv2.namedWindow("Enhanced")
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 300)

    cv2.createTrackbar("A Shift", "Controls", 50, 100, on_trackbar)
    cv2.createTrackbar("B Shift", "Controls", 50, 100, on_trackbar)
    cv2.createTrackbar("Omega", "Controls", 75, 100, on_trackbar)
    cv2.createTrackbar("CLAHE Clip", "Controls", 12, 30, on_trackbar)
    cv2.createTrackbar("Red Boost", "Controls", 30, 100, on_trackbar)

    # ----- Window for enhanced image -----
    cv2.namedWindow("Enhanced", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enhanced", 900, 900)

    # ----- Window for original image 
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 900, 900)

    cv2.createTrackbar("A Shift", "Enhanced", 50, 100, on_trackbar)
    cv2.createTrackbar("B Shift", "Enhanced", 50, 100, on_trackbar)
    cv2.createTrackbar("Omega", "Enhanced", int(OMEGA * 100), 100, on_trackbar)
    cv2.createTrackbar("CLAHE Clip", "Enhanced", int(CLAHE_CLIP * 10), 50, on_trackbar)
    cv2.createTrackbar("Red Boost", "Enhanced", RED_STRENGTH, 100, on_trackbar)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (900, 900))
    # cv2.imshow("Original", img)
    # cv2.imshow("Enhanced", enhanced)


    mode = "image"
    cap = None

    while True:

        if mode == "image":
            frame = img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (900, 900))

        enhanced = enhance_underwater(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Enhanced", enhanced)

        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == ord('v'):
            cap = cv2.VideoCapture(video_path)
            mode = "video"

        if key == ord('i'):
            mode = "image"

        # SAVE FEATURE
        if key == ord('s'):
            if mode == "image":
                cv2.imwrite("enhanced_output.png", enhanced)
                print("Saved enhanced_output.png")

            if mode == "video":
                print("Saving full enhanced video...")
                save_enhanced_video(video_path)
                print("enhanced_video.mp4 saved!")

    cv2.destroyAllWindows()


run_app(
    r"C:\\Users\\samru\\PycharmProjects\\Underwater_Image_Enhancer\\see-thru-4.jpg",
    r"C:\\Users\\samru\\PycharmProjects\\Underwater_Image_Enhancer\\video1.mp4"
)
