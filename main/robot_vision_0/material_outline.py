"""沿用 V20 的颜色分割和外轮廓规则，YOLO 仅负责给物料类别和匹配区域。"""
import math

import cv2
import numpy as np

SCALE = 0.40
SAMPLES = {
    "red": [(253,21,65),(248,17,66),(255,26,74),(253,20,69),(255,21,72),(198,59,76),(199,57,75),(177,47,61),(140,32,41),(153,38,47)],
    "yellow": [(205,213,42),(215,215,44),(207,211,32),(212,214,36),(214,214,45),(166,169,43),(158,163,22),(162,166,26),(143,143,34),(140,146,14)],
    "darkblue": [(0,78,228),(0,80,225),(0,76,220),(0,79,230),(0,68,222),(35,82,169),(37,73,129),(28,60,129),(23,62,133),(32,85,181)],
    "black": [(64,78,73),(45,65,58),(56,71,61),(56,70,65),(51,63,61),(67,73,64),(75,90,66),(67,79,70),(47,59,55),(35,43,34)],
    "lightblue": [(92,133,204),(89,129,204),(91,132,203),(93,131,203),(90,130,206),(92,130,202),(49,78,125),(40,70,110),(72,102,142),(73,107,168)],
    "green": [(96,206,87),(97,206,92),(95,207,94),(92,205,82),(93,205,92),(63,144,62),(52,135,57),(74,134,75),(90,164,86),(74,146,67)],
}
LIMITS = {"red":70,"yellow":65,"darkblue":72,"black":55,"lightblue":68,"green":70}
MIN_SATURATION = {"red":55,"yellow":55,"darkblue":60,"black":0,"lightblue":45,"green":55}


def exclusive_masks(rgb):
    """V20 同款：每个像素只归属到最近的一种物料颜色。"""
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    best = np.full(rgb.shape[:2],np.inf,np.float32)
    labels = np.full(rgb.shape[:2],-1,np.int8)
    names = list(SAMPLES)
    for index,name in enumerate(names):
        samples = np.asarray(SAMPLES[name],np.float32)
        lower = np.maximum(0,samples.min(axis=0)-35).astype(np.uint8)
        upper = np.minimum(255,samples.max(axis=0)+35).astype(np.uint8)
        possible = np.all((rgb>=lower)&(rgb<=upper),axis=2)
        if name != "black":
            possible &= hsv[:,:,1] >= MIN_SATURATION[name]
        ys,xs = np.where(possible)
        if not len(ys):
            continue
        distance = np.linalg.norm(rgb[ys,xs].astype(np.float32)[:,None,:]-samples[None,:,:],axis=2).min(axis=1)
        keep = (distance<=LIMITS[name])&(distance<best[ys,xs])
        best[ys[keep],xs[keep]] = distance[keep]
        labels[ys[keep],xs[keep]] = index
    kernel = np.ones((3,3),np.uint8)
    return {name:cv2.morphologyEx(np.where(labels==index,255,0).astype(np.uint8),cv2.MORPH_OPEN,kernel)
            for index,name in enumerate(names)}


def contour_is_material(contour):
    area = cv2.contourArea(contour)
    if area < 450:
        return False
    perimeter = cv2.arcLength(contour,True)
    if perimeter <= 0 or 4*math.pi*area/(perimeter*perimeter) < .62:
        return False
    _,radius = cv2.minEnclosingCircle(contour)
    if radius <= 0 or area/(math.pi*radius*radius) < .60:
        return False
    if len(contour) >= 5:
        axes = cv2.fitEllipse(contour)[1]
        if min(axes)/max(axes) < .65:
            return False
    return True


def bottom_outline(frame,box,name):
    """找出与 YOLO 框匹配的 V20 底部轮廓，返回颜色、外接圆、轮廓和中心 RGB。"""
    if name not in SAMPLES:
        return None
    h,w = frame.shape[:2]
    small = cv2.resize(frame,(round(w*SCALE),round(h*SCALE)),interpolation=cv2.INTER_AREA)
    masks = exclusive_masks(cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
    x1,y1,x2,y2 = box
    box_center = np.array([(x1+x2)/2,(y1+y2)/2])
    candidates = []
    contours,_ = cv2.findContours(masks[name],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for small_contour in contours:
        if cv2.contourArea(small_contour) < 450*SCALE*SCALE:
            continue
        contour = np.rint(small_contour.astype(np.float32)/SCALE).astype(np.int32)
        if not contour_is_material(contour):
            continue
        (cx,cy),radius = cv2.minEnclosingCircle(contour)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            candidates.append((float(np.linalg.norm(np.array([cx,cy])-box_center)),contour,(float(cx),float(cy)),float(radius)))
    if not candidates:
        return None
    _,contour,center,radius = min(candidates,key=lambda value:value[0])
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mask = np.zeros((h,w),np.uint8)
    cv2.circle(mask,tuple(map(round,center)),max(2,round(radius*.45)),255,-1)
    median = tuple(map(int,np.median(rgb[mask>0],axis=0)))
    return center,radius,contour,median
