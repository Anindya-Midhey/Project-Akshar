import cv2

def draw_bbox(image_path, bbox):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    xmin, ymin, xmax, ymax = bbox

    x1 = int(xmin * w)
    y1 = int(ymin * h)
    x2 = int(xmax * w)
    y2 = int(ymax * h)

    overlay = image.copy()

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    output = "highlighted.jpg"
    cv2.imwrite(output, image)

    return output