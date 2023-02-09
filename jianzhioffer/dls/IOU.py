# coding=utf-8

def compute_iou(rec1, rec2):

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = float(right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

rect1 = (661, 27, 679, 47)
# (top, left, bottom, right)
rect2 = (662, 27, 682, 47)
iou = compute_iou(rect1, rect2)
print(iou)