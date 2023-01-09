import torch
import cv2
import numpy as np

Yolov5 = torch.hub.load(
                "Yolov5",
                'custom',
                path="best.pt", 
                source='local',
                device = 'cpu',
                force_reload=True
            )
to_read = cv2.imread("toread.png")
modl = Yolov5(to_read) 
coords = modl.pandas().xyxy[0].to_dict(orient="records")
detect_annotation = np.squeeze(modl.render())

cv2.imwrite("to_read_result.png", detect_annotation)

for result in coords:
    x1 = int(result['xmin'])
    y1 = int(result['ymin'])
    x2 = int(result['xmax'])
    y2 = int(result['ymax'])
    conf = float(result['confidence'])
    if conf > 0.60 : print("Sure")
    else : print("Not Sure"); continue

    print("YES")