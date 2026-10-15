"""观察界面：等比例显示、圆心坐标、一位小数毫米值、右上角大字任务码。"""
import math
import cv2
import numpy as np

COLORS = {"red":(50,50,255),"yellow":(0,255,255),"green":(30,230,30),
          "black":(220,220,220),"lightblue":(255,210,100),"darkblue":(255,130,50)}


def text(image, value, position, scale=.5, color=(255,255,255), thickness=1):
    (width,height),baseline = cv2.getTextSize(value,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    x,y = position
    cv2.rectangle(image,(x-2,y-height-3),(x+width+2,y+baseline+2),(0,0,0),-1)
    cv2.putText(image,value,position,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def annotate(frame, items):
    image = frame.copy()
    h,w = image.shape[:2]
    origin = (w//2,h//2)
    cv2.drawMarker(image,origin,(255,255,255),cv2.MARKER_CROSS,24,2)
    text(image,"(0,0) X right / Y up",(origin[0]+10,origin[1]-10),.45)
    for item in items:
        x1,y1,x2,y2 = map(round,item["box"])
        color = COLORS.get(item["name"],(0,230,255))
        contour = item.get("contour")
        if contour is not None:
            contour = np.asarray(contour,dtype=np.int32)
            x1,y1,width,height = item["contour_box"]
            x2,y2 = x1+width,y1+height
            cv2.drawContours(image,[contour],-1,color,2)
            cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        else:
            cv2.rectangle(image,(x1,y1),(x2,y2),color,1)
        label = f"{item['name']} {item['confidence']:.2f}"
        if not item["valid"]:
            text(image,label+" / "+item["reason"],(max(2,x1),max(22,y1-7)),.6,color)
            continue
        center = tuple(map(round,item.get("display_center",item["center"])))
        # 画实际采用的顶缘或底部外接圆；框回退只画中心，避免伪装成圆拟合。
        if item.get("ellipse") is not None:
            cv2.ellipse(image,item["ellipse"],color,2,cv2.LINE_AA)
        if item.get("top_contour") is not None:
            cv2.drawContours(image,[np.asarray(item["top_contour"],dtype=np.int32)],-1,color,2,cv2.LINE_AA)
        if item.get("top_circle") is not None:
            cx,cy,radius = item["top_circle"]
            cv2.circle(image,(round(cx),round(cy)),round(radius),color,2,cv2.LINE_AA)
        if contour is not None:
            cv2.circle(image,center,4,color,-1)
            label = f"{item['name'].upper()} RGB={item['rgb']}"
        else:
            cv2.drawMarker(image,center,color,cv2.MARKER_CROSS,16,2)
        cv2.line(image,origin,center,color,1,cv2.LINE_AA)
        label += " / "+item["mode"]
        text(image,label,(max(2,x1),max(22,y1-7)),.6,color)
        prefix = "BOTTOM " if item["mode"] == "TOP CIRCLE" else ""
        text(image,f"{prefix}X {item['x']:+.1f} Y {item['y']:+.1f} mm",(max(2,x1),min(h-25,y2+18)),.45,color)
        text(image,f"D {item['distance']:.1f} Z {item['z']:.1f} mm"+(" EST" if item.get("estimated") or not item["calibrated"] else ""),
             (max(2,x1),min(h-5,y2+38)),.45,color)
    return image


def measurement_table(canvas, items):
    """底部大字摘要，避免圆旁标签在小屏缩放后难以辨认。"""
    rows = items[:8]
    if not rows:
        return
    y0 = canvas.shape[0]-65-28*len(rows)
    overlay = canvas.copy()
    cv2.rectangle(overlay,(0,y0-23),(canvas.shape[1]-1,canvas.shape[0]-55),(0,0,0),-1)
    cv2.addWeighted(overlay,.8,canvas,.2,0,canvas)
    for index,item in enumerate(rows):
        if item["valid"]:
            line = f"{item['name']}: X={item['x']:+.1f}  Y={item['y']:+.1f}  D={item['distance']:.1f}  Z={item['z']:.1f} mm"
            line += " [EST]" if item.get("estimated") or not item["calibrated"] else ""
        else:
            line = f"{item['name']}: {item['reason']} - no metric output"
        text(canvas,line,(12,y0+index*28),.65,COLORS.get(item["name"],(0,255,255)),1)


def letterbox(frame, cfg):
    w,h = cfg["width"],cfg["height"]
    scale = min(w/frame.shape[1],h/frame.shape[0])
    resized = cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
    canvas = np.zeros((h,w,3),np.uint8)
    x,y = (w-resized.shape[1])//2,(h-resized.shape[0])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
    return canvas


def task_layout(code, cfg):
    groups = code.split("+")
    lines = ["+".join(groups[:2])+"+", "+".join(groups[2:])]
    required = math.ceil(cfg["task_font_height_mm"]*cfg["height"]/cfg["physical_height_mm"])
    scale,thickness = .1,3
    font = cv2.FONT_HERSHEY_DUPLEX
    while cv2.getTextSize(lines[0],font,scale,thickness)[0][1] < required:
        scale += .05
    sizes = [cv2.getTextSize(line,font,scale,thickness) for line in lines]
    margin,gap = 20,20
    width = max(s[0][0] for s in sizes)+2*margin
    height = sum(s[0][1] for s in sizes)+gap+2*margin
    if width > cfg["width"] or height > cfg["height"]:
        raise ValueError("显示尺寸不足以完整显示设定高度的两行任务码，请核对屏幕参数")
    return lines,scale,sizes,width,height


def draw_task(canvas, code, cfg):
    if not code:
        text(canvas,"TASK: WAITING",(cfg["width"]-280,30),.7,(0,255,255),2)
        return
    lines,scale,sizes,width,height = task_layout(code,cfg)
    margin,gap = 20,20
    font = cv2.FONT_HERSHEY_DUPLEX
    for index,(line,((tw,th),_)) in enumerate(zip(lines,sizes)):
        baseline = margin+th if index == 0 else margin+sizes[0][0][1]+gap+th
        origin = (cfg["width"]-margin-tw,baseline)
        cv2.putText(canvas,line,origin,font,scale,(30,30,30),5,cv2.LINE_AA)
        cv2.putText(canvas,line,origin,font,scale,(255,255,255),3,cv2.LINE_AA)
