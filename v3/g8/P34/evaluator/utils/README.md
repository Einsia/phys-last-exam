P34 的题目测量代码位于本目录：common.py 定义结果和阈值，media.py 解码并导出媒体，models.py 加载分割模型，measurement.py 测量几何，debug.py 输出诊断图。

这些模块由上一级 `measure_backend.py` 调用；V3 入口在上一级 `evaluate.py`。运行方法见 [题目 README](../../README.md)。
