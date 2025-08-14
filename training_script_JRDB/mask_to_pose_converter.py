import numpy as np

mask = np.load('mask_000001.npy')  # Example file

object_ids = np.unique(mask)
object_ids = object_ids[object_ids != 0]  # Assuming 0 is background

def extract_bboxes(mask):
    boxes = []
    for obj_id in object_ids:
        m = (mask == obj_id)
        coords = np.argwhere(m)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        boxes.append([obj_id, y1, x1, y2, x2])
    return boxes

with open('poses.txt', 'w') as f:
    for pose in poses:
        f.write(' '.join(map(str, pose)) + '\n')
