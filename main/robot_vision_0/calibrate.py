"""标定工具：推荐棋盘格内参标定；也提供与固定距离圆形尺寸匹配的快速比例标定。"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images",help="棋盘格照片目录，至少10张不同位置/倾角的照片")
    parser.add_argument("--cols",type=int,default=9,help="棋盘格每行内角点数")
    parser.add_argument("--rows",type=int,default=6,help="棋盘格每列内角点数")
    parser.add_argument("--square-mm",type=float,default=20.,help="每格实际边长")
    parser.add_argument("--distance-mm",type=float,help="快速标定：光心到所测圆平面的实测距离")
    parser.add_argument("--diameter-px",type=float,help="快速标定：正对镜头的圆直径像素值")
    parser.add_argument("--diameter-mm",type=float,help="快速标定：该圆真实直径；顶面30、底面50")
    parser.add_argument("--width",type=int,default=1280)
    parser.add_argument("--height",type=int,default=720)
    parser.add_argument("--output",default=str(BASE/"calibration.json"))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("输出标定文件已存在，请另取名称或先备份，避免覆盖")
    if args.images:
        if args.cols<3 or args.rows<3 or args.square_mm<=0:
            parser.error("棋盘格参数无效")
        pattern = (args.cols,args.rows)
        points = np.zeros((args.cols*args.rows,3),np.float32)
        points[:,:2] = np.mgrid[0:args.cols,0:args.rows].T.reshape(-1,2)*args.square_mm
        world, image_points = [],[]
        size = None
        for path in sorted(Path(args.images).iterdir()):
            if path.suffix.lower() not in (".jpg",".jpeg",".png",".bmp"):
                continue
            image = cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            this_size = image.shape[::-1]
            if size and size != this_size:
                raise ValueError("标定照片分辨率必须一致")
            size = this_size
            found,corners = cv2.findChessboardCornersSB(image,pattern)
            if found:
                world.append(points.copy())
                image_points.append(corners)
            print(path.name,"OK" if found else "未找到完整内角点")
        if len(world)<10:
            raise ValueError(f"只有{len(world)}张有效照片，请提供至少10张覆盖全画面的棋盘格照片")
        rms,k,dist,_,_ = cv2.calibrateCamera(world,image_points,size,None,None)
        data = {"kind":"intrinsics","image_size":size,"camera_matrix":k.tolist(),
                "distortion":dist.ravel().tolist(),"rms_px":rms,"samples":len(world)}
        print(f"重投影RMS={rms:.3f}px；应再用独立距离验证毫米误差")
    else:
        values = (args.distance_mm,args.diameter_px,args.diameter_mm,args.width,args.height)
        if any(v is None or not math.isfinite(v) or v<=0 for v in values):
            parser.error("快速标定需要正数 --distance-mm --diameter-px --diameter-mm --width --height")
        ratio = args.distance_mm*args.diameter_px/(args.diameter_mm*args.width)
        data = {"kind":"scale","image_size":[args.width,args.height],"focal_width_ratio":ratio}
        print("比例标定不会校正镜头畸变；在画面中心测量，随后用不同位置和高度验证")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2),encoding="utf-8")
    print("已保存：",output)


if __name__ == "__main__":
    main()
