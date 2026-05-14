跑测试脚本如下：

```powershell
python run_all_cases.py
```

### 跑所有病例

```powershell
python run_all_cases.py
```
自动对目录下所有成对EDF/CSV病例依次处理。


### 只跑一个病例

```powershell
python run_all_cases.py --case 蔡树庄
```
或者完整名称：

```powershell
python run_all_cases.py --case H23101927F6M2_蔡树庄
```

---

### 一次跑多个病例

```powershell
python run_all_cases.py --case 刘权 --case 陈雄
```


### 只跑RR间期识别相关

```powershell
python run_all_cases.py --stage preprocess
```
主要包括ECG R 峰检测、ECG 质量窗口计算、导联 R 峰合并。


### 只跑分析阶段

预处理结果已经存在只需要跑分析阶段：

```powershell
python run_all_cases.py --stage analysis
```

单病例加`case`：

```powershell
python run_all_cases.py --case 蔡树庄 --stage analysis
```

多病例：

```powershell
python run_all_cases.py --case 刘权 --case 陈雄 --stage analysis
```

修改 AF/AFL 识别逻辑后，通常只需要重跑 `analysis`，不需要重新做 R 峰检测。

---


### 强制重跑RR间期识别：

```powershell
python run_all_cases.py --case 蔡树庄 --rerun-preprocess
```


### 分析阶段默认重新运行

分析阶段默认会重新运行并覆盖对应结果，可以用下面的指令跳过已有结果

```powershell
python run_all_cases.py --case 蔡树庄 --stage analysis --skip-existing-analysis
```

### AFL识别
当前默认使用rr2d_v2，也就是新版本的AFL识别算法。

旧版运行方式：

```powershell
python run_all_cases.py --case 蔡树庄 --stage analysis --a-flutter-mode rr1d
```

参照表：

| 参数 | 说明 |
|---|---|
| `--case 病例名` | 只运行指定病例，可重复使用 |
| `--stage all` | 跑完整流程，默认值 |
| `--stage preprocess` | 只跑预处理 |
| `--stage analysis` | 只跑分析 |
| `--rerun-preprocess` | 强制重跑预处理 |
| `--skip-existing-analysis` | 分析结果已存在时跳过 |
| `--skip-existing` | 最终评估 PDF 已存在时跳过整个病例 |
| `--a-flutter-mode rr2d_v2` | 使用新版 RR2D AFL 逻辑，默认 |
| `--a-flutter-mode rr1d` | 使用旧版一维 RR AFL 逻辑 |
| `--continue-on-error` | 某个病例失败后继续跑下一个病例 |
| `--dry-run` | 只打印命令，不实际执行 |
