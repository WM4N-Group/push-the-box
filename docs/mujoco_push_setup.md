# MuJoCo IK 锁臂推箱子第一版（macOS / Linux）

## 运行推箱子

```bash
cd "/Users/galaxy/Project/IROS 2027/go2_mobile_manipulator"
bash tools/launch_mujoco.sh --push
```

默认自动执行一轮，约 23 秒仿真时间：稳定站立 2 秒 → IK 关节轨迹 9 秒 → 到位保持 1 秒 → 前进接触 → 推箱子 0.5 m → 速度指令归零并观察 2 秒。完成或故障后冻结仿真供查看；**R 重置并重跑**，空格暂停/继续，Esc 退出。参数示例：

```bash
bash tools/launch_mujoco.sh --push --paused
bash tools/launch_mujoco.sh --push --push-distance 0.5 --speed 0.12 --box-mass 1
.venv/bin/python tools/mujoco_push_scene.py --push --headless
.venv/bin/python -m unittest discover -s tools/tests -v
```

默认报告是 `runtime/reports/box_push.json`，同名 CSV 记录 50 Hz 的机器人/箱子/TCP 位置、阶段、速度指令、六轴目标和实际角度。`--report` 可指定报告路径；`--seconds` 可限制运行时间，过短的无窗口运行会按未完成退出。任务另有 45 秒仿真超时；距离和低速度组合必须留足伸臂和接近时间。

## 控制实现

`tools/push_controller.py` 提供 IK、旧权重推理与任务状态机，`tools/mujoco_push_scene.py --push` 负责运行和报告。

- IK 在临时 MuJoCo 状态中求解，目标为机身坐标 `[0.63, 0, 0.04]` m，工具 Z 轴朝机身 +X；带关节限位、阻尼最小二乘和不可达失败检查。对关节插值路径采样检查名义自碰撞。
- 机械臂沿五次多项式关节轨迹到位；峰值目标速度不超过 25°/s、加速度不超过 50°/s²。到位后六轴目标保持不变，现有位置伺服 200/10、20 Nm 力矩限制继续生效。这里的“锁定”是**固定关节目标**，不是焊接约束，也不是每帧覆盖关节角；受重力和接触影响，实际角度有跟踪误差。
- 底盘直接使用 `arm_moving_clock/exported/policy.onnx`，保持旧 ABI：机身速度/重力、五维指令、腿状态、前次动作、步态时钟、机械臂角度，60 维×5 帧历史。旧策略机械臂参考角是零，不能换成标准版 HOME_ARM。策略 50 Hz、物理 200 Hz、腿部 PD 20/1.5；未重新训练权重。
- 前向速度默认 0.12 m/s，附加横向和航向反馈，使夹爪对齐箱子且机身面向 +X；速度指令平滑变化。位置来自仿真真值，尚未接视觉或状态估计。
- 实际使用原夹爪两指碰撞面推箱子；基座和箱子自由运动，无附加推板，无对箱子或机器人施加的外力，无运行中基座/箱子位姿覆盖。
- 通过夹指—箱子实际接触力确认进入 PUSH；达到箱子沿 +X 的目标位移后减速归零，观察 2 秒再确认成功。成功还要求累计有效接触至少 0.2 秒。
- 失稳、臂跟踪异常、超过 5 N 的非预期接触、箱子倾斜超过 0.4 rad、总推力超过 100 N、丢失接触、横漂超过 0.15 m 或超时会使本轮 FAILED 并停止物理步进。阈值用于本仿真试验，不是真机安全控制器。

## 实测范围与限制

同一平地场景，箱子边长 0.4 m，地面滑动摩擦系数 0.8、箱子 0.6（实际接触组合由 MuJoCo 决定），默认速度 0.12 m/s，目标 0.5 m：

| 箱子质量 / 初始近侧距离 | 结果 | 箱子前移 | 横漂 | 说明 |
|---|---|---|---|---|
| 1 kg / 1.0 m | 通过 | 0.525 m | -0.016 m | 23.04 秒完成 |
| 0.5 kg / 1.2 m | 通过 | 0.521 m | -0.012 m | 23.82 秒完成 |
| 2 kg / 1.0 m | 失败停止 | 0.144 m | -0.003 m | 18.47 秒箱子倾斜超过 0.4 rad |

默认 1 kg 测试最大机身倾斜 0.0484 rad（2.77°），最大推力约 33.84 N，最大臂关节跟踪误差 0.0611 rad（3.50°），无 MuJoCo 警告，无超过 5 N 的非预期接触。名义 IK 位置残差约 0.12 mm，仅指运动学解，不是带负载实际 TCP 误差。原生窗口也已运行验证。三个回归测试覆盖不可达 IK、完整物理推动与关节目标保持/重置，以及跌倒时故障锁定。

**这是一版可运行的混合控制基线，不是新训练的推箱子网络。** 旧权重原本对应 Piper L，本次在标准 PiPER 上完成上述有限场景的经验验证；不能推断任意质量、摩擦、姿态或真机条件下都可靠。2 kg 的失败表明当前固定接触高度和刚性保持方案仍有倾覆边界，后续可调低接触点、改末端接触面或增加力反馈。

## 保留的站立场景

- 独立 `.venv`：Python 3.9.6、MuJoCo 3.2.3、NumPy 1.26.4、ONNX Runtime 1.19.2、PyYAML 6.0.2。
- Go2 + **标准 PiPER** 模型、地面、可自由平移和转动的箱子，开启实际碰撞和摩擦。
- 底盘起点 `(0, 0, 0.32)` m，四元数 `(1, 0, 0, 0)`，面向世界 +X；站立 PD 模式落地后机身高度约 0.259 m，推箱模式的策略站立约 0.294 m。
- 箱子边长 0.4 m、质量 1 kg，初始中心 `(1.2, 0, 0.2)` m；近侧面距初始机身原点 1 m。这个距离不是距鼻端或夹爪的净间隙。
- 站立腿部 PD 60/5；机械臂保持 `[0, 1, -1.2, 0, 0.2, 0]` rad，使用原标准模型的伺服参数。机器人基座和箱子均未固定。
- 箱子近侧面有蓝色参考点，离地约 0.3 m；不带 `--push` 时仅保持站立。

## 打开窗口

```bash
cd "/Users/galaxy/Project/IROS 2027/go2_mobile_manipulator"
bash tools/launch_mujoco.sh
```

无需激活环境或启动 ROS、Docker、Isaac Lab。脚本在 macOS 使用 `mjpython` 的 Cocoa 启动器，并通过显式 Python 调用解决当前项目路径含空格的问题；不要直接执行 `.venv/bin/mjpython`。

窗口中按空格暂停/继续，按 **R** 重新初始化机器人和箱子，按 Esc 退出。鼠标左键拖动旋转、右键拖动平移、滚轮缩放。应使用 R 进行本任务重置。界面中的关节/Control 滑块会被站立控制循环覆盖。

```bash
# 打开后暂停，方便观察初始姿态
bash tools/launch_mujoco.sh --paused

# 改变箱子近侧面距离和质量
bash tools/launch_mujoco.sh --box-distance 1.2 --box-mass 2
```

`--box-distance` 最小 0.8 m；它只改变箱子位置。箱子尺寸、摩擦及蓝色参考点位于 `third_party/locomotion_policy/deploy/mujoco/models/scene_push_box.xml`。机器人资产来自同目录下的 `go2_piper_standard.xml`，未改动原资产。

## 依赖重建及检查

另一个机器/目录需要重新创建虚拟环境，不要搬运 `.venv`：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mujoco.txt
.venv/bin/python -m pip check
.venv/bin/python tools/mujoco_push_scene.py --headless --seconds 30 --check-policy
```

本次 macOS ARM64 安装的完整版本快照见 `requirements-mujoco-macos.lock.txt`，便于同平台复现。主仓库 `requirements.txt` 面向 Ubuntu/Foxy 真机部署，不用于这里的独立场景；既有 `play_mujoco.py` 训练策略评估入口还有 PyTorch 等依赖，本次入口不需要它们。

无窗口检查报告保存在 `runtime/reports/mujoco_push_scene.json`。2026-11-12 本机 30 秒结果：最低机身高度 0.258687 m，最大倾斜 0.020134 rad，箱子沉降约 0.108 mm，无 MuJoCo 警告，无超过 5 N 的非预期接触；原生窗口已实际打开并检查机器人和箱子显示。

`--check-policy` 仍可单独检查 ONNX 加载及输入输出形状；完整控制验收使用 `--push --headless`。
