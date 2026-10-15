"""YOLO区域内精细圆拟合与针孔测量。要求相机光轴竖直、物料直立。

X向右、Y向上，原点为图像几何中心在目标平面上的投影；Z为光心到目标平面距离。
仅保存一位小数是输出格式，不代表0.1mm的真实精度。
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Circle:
    x: float
    y: float
    radius: float
    normalized_radius: float
    error: float
    ellipse: tuple
    contour: np.ndarray


class CameraGeometry:
    def __init__(self, settings, base):
        self.settings = settings
        self.calibration = None
        self.size = None
        self.maps = None
        path = Path(base) / settings["calibration"]
        if path.exists():
            self.calibration = json.loads(path.read_text(encoding="utf-8"))
            if self.calibration.get("kind") not in ("intrinsics", "scale"):
                raise ValueError("不支持的标定文件格式")

    @property
    def calibrated(self):
        return self.calibration is not None

    def prepare(self, frame):
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self.size = (w, h)
            if self.calibration:
                cw, ch = self.calibration["image_size"]
                if abs(w / h - cw / ch) > 0.01:
                    raise ValueError("标定与采集宽高比不同，请在当前分辨率重新标定")
                if self.calibration["kind"] == "intrinsics":
                    k = np.asarray(self.calibration["camera_matrix"], dtype=float).copy()
                    k[0, :] *= w / cw
                    k[1, :] *= h / ch
                    dist = np.asarray(self.calibration["distortion"], dtype=float)
                    self.maps = cv2.initUndistortRectifyMap(k, dist, None, k, (w, h), cv2.CV_32FC1)
                else:
                    f = w * float(self.calibration["focal_width_ratio"])
                    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]])
            else:
                f = w / (2 * math.tan(math.radians(self.settings["horizontal_fov_deg"]) / 2))
                k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]])
            if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
                raise ValueError("无效相机内参")
            self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        return cv2.remap(frame, *self.maps, cv2.INTER_LINEAR) if self.maps else frame

    def position(self, center, depth):
        w, h = self.size
        x = (center[0] - w / 2) * depth / self.fx
        y = -(center[1] - h / 2) * depth / self.fy
        return float(x), float(y), float(math.hypot(x, y))

    def bottom_from_top(self, center, top_depth):
        # 竖直物料顶/底圆心具有相同的相机坐标X/Y，而在图像中的投影不同。
        # 不能把顶面像素中心原封不动当成底面像素中心。
        bottom_depth = top_depth + self.settings["material_height_mm"]
        x = self.cx + (center[0] - self.cx) * top_depth / bottom_depth
        y = self.cy + (center[1] - self.cy) * top_depth / bottom_depth
        return (float(x), float(y)), float(bottom_depth)


def circle_candidates(roi, offset, camera, min_axis_ratio=None, max_fit_error=None):
    """提取圆或椭圆边缘，并以椭圆方程残差验证拟合质量。"""
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    masks = [cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
             cv2.Canny(gray, 35, 110)]
    # 顶面与侧面可能灰度相同但颜色不同；逐通道边缘补充灰度边缘。
    masks.extend(cv2.Canny(cv2.GaussianBlur(channel,(5,5),0),35,110) for channel in cv2.split(roi))
    candidates = []
    ox, oy = offset
    cfg = camera.settings
    min_axis_ratio = cfg["min_axis_ratio"] if min_axis_ratio is None else min_axis_ratio
    max_fit_error = cfg["max_fit_error"] if max_fit_error is None else max_fit_error
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 20 or cv2.contourArea(contour) < math.pi * cfg["min_radius_px"] ** 2 * 0.7:
                continue
            points = contour.reshape(-1, 2).astype(np.float32) + [ox, oy]
            ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
            (x, y), axes, angle = ellipse
            major,minor = max(axes)/2,min(axes)/2
            if minor / major < min_axis_ratio or minor < cfg["min_radius_px"]:
                continue
            # 拒绝碰到ROI边界的轮廓，避免把被截断目标当成完整圆。
            local = contour.reshape(-1, 2)
            if local[:, 0].min() < 2 or local[:, 1].min() < 2 or local[:, 0].max() >= roi.shape[1]-2 or local[:, 1].max() >= roi.shape[0]-2:
                continue
            radians = math.radians(angle)
            dx,dy = points[:,0]-x,points[:,1]-y
            u = dx*math.cos(radians)+dy*math.sin(radians)
            v = -dx*math.sin(radians)+dy*math.cos(radians)
            residual = np.abs(np.sqrt((u/(axes[0]/2))**2+(v/(axes[1]/2))**2)-1)
            inliers = residual <= .12
            if inliers.mean() < .55 or inliers.sum() < 20:
                continue
            ellipse = cv2.fitEllipse(points[inliers].astype(np.float32).reshape(-1,1,2))
            (x,y),axes,angle = ellipse
            major,minor = max(axes)/2,min(axes)/2
            if minor / major < min_axis_ratio or minor < cfg["min_radius_px"]:
                continue
            radians = math.radians(angle)
            dx,dy = points[inliers,0]-x,points[inliers,1]-y
            u = dx*math.cos(radians)+dy*math.sin(radians)
            v = -dx*math.sin(radians)+dy*math.cos(radians)
            error = float(np.sqrt(np.mean((np.sqrt((u/(axes[0]/2))**2+(v/(axes[1]/2))**2)-1)**2)))
            if error > max_fit_error:
                continue
            radius = math.sqrt(major*minor)
            nr = math.sqrt((major/camera.fx)*(minor/camera.fy))
            candidate = Circle(x, y, radius, nr, error, ellipse,
                               np.rint(points[inliers]).astype(np.int32).reshape(-1,1,2))
            duplicate = next((c for c in candidates if math.hypot(c.x-x,c.y-y)<2 and abs(c.radius-radius)<1.5), None)
            if duplicate is None:
                candidates.append(candidate)
            elif error < duplicate.error:
                candidates[candidates.index(duplicate)] = candidate
    return sorted(candidates, key=lambda c: c.radius)


def top_circle_candidates(roi, offset, camera):
    """只接受圆形顶部边缘；不把倾斜、拉伸的边缘按椭圆测量。"""
    gray = cv2.GaussianBlur(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(5,5),0)
    masks = [cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1],
             cv2.Canny(gray,35,110)]
    masks.extend(cv2.Canny(cv2.GaussianBlur(channel,(5,5),0),35,110) for channel in cv2.split(roi))
    ox,oy = offset
    cfg = camera.settings
    candidates = []
    for mask in masks:
        contours,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 20 or cv2.contourArea(contour) < math.pi*cfg["min_radius_px"]**2*.7:
                continue
            perimeter = cv2.arcLength(contour,True)
            if perimeter <= 0 or 4*math.pi*cv2.contourArea(contour)/(perimeter*perimeter) < .72:
                continue
            local = contour.reshape(-1,2)
            if local[:,0].min()<2 or local[:,1].min()<2 or local[:,0].max()>=roi.shape[1]-2 or local[:,1].max()>=roi.shape[0]-2:
                continue
            points = local.astype(np.float32)+[ox,oy]
            (x,y),radius = cv2.minEnclosingCircle(points.astype(np.float32).reshape(-1,1,2))
            if radius < cfg["min_radius_px"]:
                continue
            normalized = (points-[x,y])/[camera.fx,camera.fy]
            distances = np.linalg.norm(normalized,axis=1)
            nr = float(np.median(distances))
            error = float(np.sqrt(np.mean((distances-nr)**2))/nr)
            if error > cfg["max_fit_error"]:
                continue
            candidate = Circle(x,y,radius,nr,error,None,np.rint(points).astype(np.int32).reshape(-1,1,2))
            duplicate = next((c for c in candidates if math.hypot(c.x-x,c.y-y)<2 and abs(c.radius-radius)<1.5),None)
            if duplicate is None or error < duplicate.error:
                if duplicate is not None:
                    candidates[candidates.index(duplicate)] = candidate
                else:
                    candidates.append(candidate)
    return sorted(candidates,key=lambda candidate:candidate.radius)


def find_ring(circles, cfg):
    """用多圈尺寸关系确认最内侧边缘，拒绝仅凭一个圆形杂物就输出target。"""
    first = cfg["target_first_outer_diameter_mm"]
    line = cfg["target_line_width_mm"]
    outer_diameters = [first + d for d in (0, 5, 12, 22, 32, 42)]
    diameters = sorted(d for outer in outer_diameters for d in (outer-2*line, outer))
    inner_diameter = diameters[0]
    best = None
    for inner in circles:
        matched = {0}
        errors = []
        for other in circles:
            if math.hypot(inner.x-other.x, inner.y-other.y) > max(3.0, inner.radius*0.10):
                continue
            estimate = other.radius / inner.radius * inner_diameter
            index = min(range(len(diameters)), key=lambda i: abs(diameters[i]-estimate))
            # 1.2mm的比例容差，避免把大圈错误认成最内圈。
            if abs(diameters[index]-estimate) < 1.2:
                matched.add(index)
                errors.append(abs(diameters[index]-estimate))
        # 至少三个不同印刷圈的边缘，并包含最内圈的外边缘。
        if 1 not in matched or len({i//2 for i in matched}) < 3:
            continue
        score = len(matched) - sum(errors) / max(len(errors), 1)
        if best is None or score > best[0]:
            best = (score, inner)
    return best[1] if best else None


def find_top(circles, box, camera):
    """优先找30mm顶面椭圆，并用50mm底面候选验证其高度关系。"""
    x1,y1,x2,y2 = box
    short = min(x2-x1,y2-y1)
    center = ((x1+x2)/2,(y1+y2)/2)
    possible = [circle for circle in circles
                if short*.14 <= circle.radius <= short*.43
                and math.hypot(circle.x-center[0],circle.y-center[1]) <= short*.48]
    matches = []
    for top in possible:
        top_depth = camera.settings["material_top_diameter_mm"]/(2*top.normalized_radius)
        bottom_center,bottom_depth = camera.bottom_from_top((top.x,top.y),top_depth)
        for bottom in circles:
            if bottom is top:
                continue
            expected = camera.settings["material_bottom_diameter_mm"]/(2*bottom.normalized_radius)
            if abs(expected-bottom_depth) > max(12.,bottom_depth*.07):
                continue
            error = math.hypot(bottom.x-bottom_center[0],bottom.y-bottom_center[1])
            if error <= max(5.,top.radius*.25):
                matches.append((error,top,top_depth,bottom_center,bottom_depth))
    if matches:
        _,top,top_depth,bottom_center,bottom_depth = min(matches,key=lambda value:value[0])
        return top,top_depth,bottom_center,bottom_depth
    if len(possible) == 1:
        top = possible[0]
        top_depth = camera.settings["material_top_diameter_mm"]/(2*top.normalized_radius)
        bottom_center,bottom_depth = camera.bottom_from_top((top.x,top.y),top_depth)
        return top,top_depth,bottom_center,bottom_depth
    return None


def measure(frame, box, name, confidence, camera):
    h, w = frame.shape[:2]
    box = tuple(map(float, box))
    x1, y1, x2, y2 = box
    item = {"name": name, "confidence": confidence, "box": box, "valid": False,
            "reason": "NO CIRCLE", "calibrated": camera.calibrated}
    if not all(math.isfinite(v) for v in box) or x2 <= x1 or y2 <= y1:
        item["reason"] = "INVALID BOX"
        return item
    if name.startswith("target"):
        number = name.removeprefix("target")
        item["number"] = int(number) if number in ("1", "2", "3") else None
    def fallback(reason):
        # 圆缘不可见时仍输出框中心；直径尺度来自物料底面或圆环最外圈。
        left, top, right, bottom = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
        if right <= left or bottom <= top:
            item["reason"] = "OUTSIDE FRAME"
            return item
        center = ((left+right)/2, (top+bottom)/2)
        diameter = (camera.settings.get("target_outer_diameter_mm",95.0) if name.startswith("target")
                    else camera.settings["material_bottom_diameter_mm"])
        depth = diameter*math.sqrt(camera.fx*camera.fy)/min(right-left,bottom-top)
        x,y,distance = camera.position(center,depth)
        item.update(valid=True,reason=reason,center=center,x=x,y=y,z=depth,distance=distance,
                    ellipse=None,mode="BOX EST",estimated=True)
        return item
    if x1 <= 2 or y1 <= 2 or x2 >= w-2 or y2 >= h-2:
        return fallback("CLIPPED")
    pad = max(8, round(min(x2-x1,y2-y1)*0.12))
    left, top = max(0,int(x1)-pad), max(0,int(y1)-pad)
    right, bottom = min(w,math.ceil(x2)+pad), min(h,math.ceil(y2)+pad)
    is_target = name.startswith("target")
    if is_target:
        circles = circle_candidates(frame[top:bottom,left:right], (left,top), camera)
        circle = find_ring(circles, camera.settings)
        mode = "INNER EDGE"
        if circle is None:
            return fallback("NO CONCENTRIC RINGS")
        diameter = camera.settings["target_first_outer_diameter_mm"] - 2*camera.settings["target_line_width_mm"]
        depth = diameter / (2*circle.normalized_radius)
        center = (circle.x, circle.y)
        number = name.removeprefix("target")
        item["number"] = int(number) if number in ("1","2","3") else None
    else:
        circles = top_circle_candidates(frame[top:bottom,left:right],(left,top),camera)
        top_measurement = find_top(circles,box,camera)
        if top_measurement is not None:
            top_circle,top_depth,center,depth = top_measurement
            item["display_center"] = (top_circle.x,top_circle.y)
            item["top_contour"] = top_circle.contour.tolist()
            item["top_circle"] = (top_circle.x,top_circle.y,top_circle.radius)
            item["top_depth"] = top_depth
            mode = "TOP CIRCLE"
            ellipse = None
        else:
            from material_outline import bottom_outline
            outline = bottom_outline(frame, box, name)
            if outline is None:
                return fallback("NO TOP OR BOTTOM OUTLINE")
            center, radius, contour, rgb = outline
            depth = camera.settings["material_bottom_diameter_mm"]*math.sqrt(camera.fx*camera.fy)/(2*radius)
            item["contour"] = contour.tolist()
            item["contour_box"] = cv2.boundingRect(contour)
            item["rgb"] = rgb
            item["radius_px"] = radius
            mode = "BOTTOM OUTLINE"
            ellipse = (center,(2*radius,2*radius),0.0)
    if not 0 < depth <= camera.settings["max_depth_mm"]:
        return fallback("DEPTH OUT OF RANGE")
    x,y,distance = camera.position(center, depth)
    item.update(valid=True, reason="", center=center, x=x, y=y, z=depth, distance=distance,
                ellipse=circle.ellipse if is_target else ellipse, mode=mode)
    return item
