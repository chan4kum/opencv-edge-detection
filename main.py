import argparse
import cv2
import numpy as np

WIN = "Edges"


def nothing(_):
    pass


def edges(gray, lo, hi):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, lo, hi)
    sx, sy = cv2.Sobel(blur, cv2.CV_64F, 1, 0), cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    lap = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
    return canny, sobel, lap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    args = p.parse_args()
    still = cv2.imread(args.image) if args.image else None
    if args.image and still is None:
        raise SystemExit(f"Could not read {args.image}")
    cap = None if still is not None else cv2.VideoCapture(0)

    cv2.namedWindow(WIN)
    cv2.createTrackbar("Canny low", WIN, 50, 255, nothing)
    cv2.createTrackbar("Canny high", WIN, 150, 255, nothing)

    while True:
        if still is not None:
            frame = still.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break
        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny, sobel, lap = edges(gray, cv2.getTrackbarPos("Canny low", WIN), cv2.getTrackbarPos("Canny high", WIN))
        top = np.hstack([gray, canny])
        bottom = np.hstack([sobel, lap])
        cv2.imshow(WIN, np.vstack([top, bottom]))  # gray | canny  /  sobel | laplacian
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
