def normalize_boxes(data, img_w, img_h):
    normalized = []

    for item in data["words"]:
        x = item["bbox"]["x"]
        y = item["bbox"]["y"]
        w = item["bbox"]["width"]
        h = item["bbox"]["height"]

        xmin = x / img_w
        ymin = y / img_h
        xmax = (x + w) / img_w
        ymax = (y + h) / img_h

        normalized.append({
            "text": item["text"],
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return normalized