# 圆环与物料视觉检测

双击 `start.cmd` 启动。Windows 默认使用本机 `D:\anaconda3\envs\yolov8\python.exe`，
摄像头0、COM3；Linux 使用 `/dev/camera1`、`/dev/ttyUSB0`。
先打开串口和相机预览，再由后台线程加载模型。按 Q、Esc 或关闭窗口退出。
本程序不安装开机自启，不在启动时下载模型、安装依赖或扫描设备。

## 配置

编辑 `config.json`。默认权重 `../runs/detect/material_1/weights/best.pt` 已检查类别顺序：
`red, lightblue, darkblue, green, yellow, black, target1, target2, target3`。
旧权重若类别顺序不同，程序会报错，不能靠修改配置掩盖错误标签。
`target1/2/3` 都属于圆环 target，数字用于串口编号。
如果未来权重只有一个 `target` 类，也能画内圆和测量，但没有编号时不发送圆环坐标包。

Windows 双击启动也会打开串口；没有电控时可使用：

```powershell
.\start.cmd --no-serial
```

Linux/Jetson 中，在已经装好适配设备的 PyTorch、Ultralytics、OpenCV、NumPy 的环境运行：

```bash
python3 -m pip install pyserial
chmod +x start.sh
./start.sh
```

支持文件管理器把 `start.sh` 设置为可执行后“在终端运行”；无需 sudo。
如使用虚拟环境，可以设置 `ROBOT_PYTHON` 为该环境 Python 的绝对路径。
不要为安装本程序盲目替换 Jetson 的 CUDA/PyTorch。`requirements.txt` 仅列依赖，不固定设备专用版本。
若单独复制此目录到设备，将权重复制至目录内自建的 `models/best.pt`，再修改 `model`。
串口不可用会打印原因，视觉预览继续运行。相机不输出 MJPG 时明确报错。

## 测量定义与使用条件

- 相机光轴要求竖直向下，物料直立。倾斜相机需要额外外参/平面姿态标定，本程序不把光轴距离冒充倾斜场景的竖直高度。
- 原点是相机画面中心在相应目标平面上的投影；X向右、Y向上，Z是相机光心到平面的正距离。
- D是平面距离 `sqrt(X²+Y²)`，不是空间斜距；所有显示值及串口浮点值均保留一位小数。
- 图像使用等比例缩放和黑边。测量基于原相机分辨率，界面缩放不改变坐标。
- 物料优先拟合30mm顶面圆。窗口圆线和十字显示顶面圆心；程序把该结果换算到底部圆心，
  串口的 X/Y/D/Z 始终是底部圆心和底部平面的值。顶面圆不可辨认时，按 V20 的颜色外轮廓和50mm底部外缘计算。
- 圆环：最内圈印刷线外径53mm、线宽1.5mm。程序拟合最内侧边缘，使用直径50mm，
  并用至少三个印刷圈的同心和尺寸关系校验。若你实际打印尺寸不同，先实测并更新配置。
- 物料显示 `TOP CIRCLE` 时，画顶面圆但标注 `BOTTOM X/Y`；显示 `BOTTOM OUTLINE` 时，画底部外轮廓和外接圆。Z是相机到底部平面的距离，D是底圆中心距画面中心的平面距离。
- 底部轮廓不可辨认，或目标贴边时，输出可见检测框中心，显示 `BOX EST`。圆环几何验证失败也使用这一回退。
  框估算按物料底径50mm、圆环最外径95mm换算毫米坐标与高度；估算值同样发送。没有YOLO检测框时不虚构目标。
- 纯目标检测框本身不能提供精确圆心与毫米距离。YOLO负责类别/区域，OpenCV负责区域内几何测量。

### 标定

未提供相机实测标定时，默认按水平60度视场角估算。界面标记 `EST`，
`serial.require_calibration=false` 默认发送估算坐标；轮廓/框回退即使已有标定也标记 `EST`。
把小数显示成0.1mm不代表实际误差是0.1mm，必须在真实安装姿态/距离验证。

推荐采集至少10张覆盖画面不同位置和倾角的棋盘格照片：

```bash
python calibrate.py --images ./checkerboard --cols 9 --rows 6 --square-mm 20
```

cols/rows为内角点数，square-mm为实际格长。生成 `calibration.json` 后重启，程序按内参去畸变。
如果相机是自动对焦，应固定焦距后采集标定；改变焦距、分辨率裁切或光学配置后重新标定。

快速单点比例标定（示例数值必须替换成实测值）：

```bash
python calibrate.py --distance-mm 300 --diameter-px 110 --diameter-mm 30 --width 1280 --height 720
```

这里 distance-mm 是相机光心到所测30mm顶面圆的距离，不是到底部的距离。
快速标定仅修正焦距比例，不能校正畸变；完成后应在多个位置和高度独立验证。
如果需要禁止未标定坐标发送，可将 `serial.require_calibration` 改为 true。

## 串口与任务码

115200、8N1。协议不变：

| 内容             | 格式                                               |
| ---------------- | -------------------------------------------------- |
| 物料，17字节     | `AA BB` + 颜色1字节 + X/Y/Z各float32小端 + `CC DD` |
| 圆环，13字节     | `BB AA` + 编号1字节 + X/Y各float32小端 + `DD CC`   |
| 接收任务，23字节 | ASCII `CSCS156+123+516+231GOGO`                    |

通信颜色：red=1、yellow=2、darkblue=3、green=4、black=5、lightblue=6。
它与YOLO从0开始的类别编号不同，代码已单独映射。
圆环Z在界面/终端输出，不追加到旧包。圆环默认每0.2秒、物料每1秒发送。
无效/过期检测不发送，旧协议没有“目标消失”包，电控应按自己的接收超时规则处理。

任务码连续两个合法帧相同才锁定，损坏帧清零确认计数，锁定后不再变化。
窗口提交任务码画面后开始每50ms发送一次ASCII `F`，持续2秒，不重复启动确认。
默认任务码由外部二维码模块/电控经串口送入，与参考协议一致。
也支持 `task_code.source="camera"`：相机后台扫码，连续两次相同后显示；
此模式不冒充电控双帧接收，不向电控发送F或自动发送新的任务码协议。

右上角以紧凑的浅色字体、半透明黑底，两行显示完整任务码；默认1280×800显示器物理高66mm，字高约6mm。
可在 `config.json` 中调整 `task_font_height_mm`；请将 `display.physical_height_mm` 设置为实际屏幕高度。
任务执行完成状态未在既有接收协议中定义，本程序不伪造已完成状态。

## 离线检查

```bash
python selftest.py
python app.py --check
python app.py --image path/to/test.jpg --output output/test.jpg
python app.py --image path/to/test.jpg --task-code 156+123+516+231 --output output/ui.jpg
```

离线输出叠加图和同名JSON，不打开摄像头或串口，不修改模型。
首次推理包含CUDA/模型初始化时间；稳定运行帧率要在目标设备上测量。
检测结果与其原帧绑定，避免把旧坐标画在新帧上；超过0.4秒的结果不发送。
若设备推理超过此值，先优化输入尺寸/推理后端，再根据电控可接受延迟调整阈值。
数字圆环编号来自YOLO类别；几何验证只校验环形，不额外用旧数字模板启动识别。

## 未来第二摄像头

采集器、Detector 和推理线程均为实例，不共享当前帧和坐标。后续黑色障碍物模型可单独实例化，
复用串口单线程发送入口即可。当前没有打开第二个摄像头、加载第二套权重或增加通信字段，
不影响当前启动时间。障碍物包格式届时需要与电控另行约定。
