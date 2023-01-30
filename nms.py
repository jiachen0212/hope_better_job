# nms 
# coding=utf-8
import numpy as np 


'''
1. 置信度阈值, 和 iou阈值
(低于置信度阈值的box直接丢掉, 也就是分类score太低的box直接丢弃.)
(对于剩余的box, 按照score分数降序, 最高分box分别和剩下的各个box算iou(前提是这些box的类别一致, 不指向同一类的box无需算iou), 
iou值高于iou阈值的box被剔除.  认为他们是冗余的, 和最高分box交集太多可以剔除)

'''

def IOU(cur_box, boxes_list):
	# x1, y1, x2, y2
	cur_box_area = (cur_box[2]-cur_box[0]+1)*(cur_box[3]-cur_box[1]+1)
	x1_boxes_list = [box_[0] for box_ in boxes_list]
	y1_boxes_list = [box_[1] for box_ in boxes_list]
	x2_boxes_list = [box_[2] for box_ in boxes_list]
	y2_boxes_list = [box_[3] for box_ in boxes_list]
	# 找出x1的最大x1max, x2的最小x2min, y1的最大y1max, y2的最小y2min
	# 相交的面积: (x2min-x1max)*(y2min-y1max)
	x1max = np.maximum(cur_box[0], x1_boxes_list)
	x2min = np.minimum(cur_box[2], x2_boxes_list)
	y1max = np.maximum(cur_box[1], y1_boxes_list)
	y2min = np.minimum(cur_box[3], y2_boxes_list)
	boxes_list_areas = [(box_[2]-box_[0]+1)*(box_[3]-box_[1]+1) for box_ in boxes_list]
	lens = len(boxes_list)
	# iou = inter / (cur_area + box_area - inter)
	inter = [(max(x2min[i]-x1max[i], 0)+1)*(max(y2min[i]-y1max[i], 0)+1) for i in range(lens)]

	ious = [inter[i] / (cur_box_area+boxes_list_areas[i]-inter[i]) for i in range(lens)]
	return ious 


def nms(boxes, conf_thres=0.7, iou_thres=0.4):
	# 高于置信度阈值的box list, 和, nms后的box list
	bbox_upper_conf_thres = []
	nms_res_bbox = []
	# x: [x1,y1,x2,y2,cls,score] 按照分数降序排序boxes
	box_sorted = sorted(boxes, reverse=True, key=lambda x:x[5])
	bbox_upper_conf_thres = [box_ for box_ in box_sorted if box_[5] >= conf_thres]

	while bbox_upper_conf_thres:
		cur_box = bbox_upper_conf_thres.pop(0)
		# 先把当前score最大, 且一会要和其他box算iou的box, 放入nms结果list
		nms_res_bbox.append(cur_box)
		# 选择和cur_box指向的cls一致的box, 计算iou
		need_iou_boxes = [bbox for bbox in bbox_upper_conf_thres if bbox[4] == cur_box[4]]
		ious = IOU(cur_box, need_iou_boxes)
	
		# 开始剔除那些iou过大的box
		remove_iou_index = [ind for ind in range(len(ious)) if ious[ind] >= iou_thres]
		while remove_iou_index:
			remove_box = need_iou_boxes[remove_iou_index.pop()]
			bbox_upper_conf_thres.remove(remove_box)

	# 遍历完了所有的boxes, nms_res_bbox也把该留下的都留下了
	return nms_res_bbox 


if __name__ == "__main__":

	import cv2
	img = cv2.imread('./IMG_6831.JPG')
	img1 = img.copy()
	h,w = img.shape[:2]
	cv2.rectangle(img, (0,90), (690,672), (255, 0, 255), 1)
	cv2.rectangle(img, (0,70), (600,600), (0, 0, 255), 1)
	cv2.rectangle(img, (0,85), (710,690), (255, 0, 0), 1)
	# cv2.imshow('', img)
	# cv2.waitKey(5000)


	boxes = [[0,90,690, 672,'fugui', 0.99],[0,70,600,600,'fugui', 0.95],[0,85,710,690,'fugui',0.9]]
	nms_res = nms(boxes)
	cv2.rectangle(img1, (nms_res[0][0],nms_res[0][1]), (nms_res[0][2],nms_res[0][3]), (255, 255, 255), 1)
	compare = np.concatenate((img, img1), axis=1)
	# cv2.imshow('', compare)
	# cv2.waitKey(5000)
	cv2.imwrite('./nms.jpg', compare)







        