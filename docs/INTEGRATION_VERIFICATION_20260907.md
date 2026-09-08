# 通用配准 2.0.0 接口迁移与构建核验

核验时间：2026-09-07T18:14:50+08:00

## 结论

下颌项目已经改用 `general-model-registration==2.0.0` 的版本化公共接口，运行时不再向 `sys.path` 注入相邻通用项目源码。核心配准、阶段复核、多区域髁突选区、旧项目适配和真实六文件流程均已通过核验。本次没有调整配准配置、质量门槛、候选排序、矩阵方向或髁突中心定义，也没有构建 EXE。

## 依赖与独立安装

- 通用 wheel：`general_model_registration-2.0.0-py3-none-any.whl`
- 通用 wheel SHA-256：`93eff461b4c4a44ec315c0787efd5fbca8ec7f63343be7efd89ded4b53b38c34`
- 测试解释器：项目隔离虚拟环境中的 Python 3.12
- 通用接口导入位置：`.venv-integration\Lib\site-packages\auto_alignment\integration\__init__.py`
- 下颌 wheel 导入位置：`.venv-integration\Lib\site-packages\mandible_registration\__init__.py`
- `INTEGRATION_API_VERSION == 1`
- `GENERAL_MODEL_REGISTRATION_VERSION == 2.0.0`
- `pip check`：通过
- 从仓库外使用 `python -I` 导入：通过；`sys.path` 中没有相邻 `general_model_registration\src`

## 代码迁移

- `core_bridge.py`：从父目录源码注入改为公共 API 版本核验。
- `workflow.py`、`viewer.py`：改用 `auto_alignment.integration.core`。
- `stage_review.py`：改用公共复核清单构造、校验和读取接口，保留 `stage_key`、`transformation`、归档路径和旧 `T_DELTA/target.stl` 兜底。
- `condyle_selection.py`：改用 `MultiRegionSelectionViewer`；业务 profile 与 `.regions.v1.json` UI 状态分开。
- `condyles.py`：改用公开 `encode_face_ranges`，面积加权中心算法不变。
- 测试和核验脚本：不再导入通用项目私有模块或私有符号。
- `pyproject.toml`：声明精确依赖 `general-model-registration==2.0.0` 和开发测试依赖。

## 自动测试

- 下颌项目：84 项通过；仅有 3 条 VTK/NumPy 2.5 弃用警告。
- 通用项目公共接口专项：14 项通过。
- 通用项目全套：98 项通过。测试结束后的 pytest 临时目录清理受到沙箱权限限制并打印 `PermissionError`，但测试进程退出码为 0。
- 编译检查：`src`、`tests`、`scripts` 全部通过。
- 私有耦合审计：产品代码、测试及核验脚本未发现旧 `auto_alignment.config/mesh_io/registration/result_viewer/model_viewer/mesh_selection` 导入，也未发现通用项目源码路径注入。

通用项目全套测试在更早的一次运行中曾出现一次约 `6.36e-9` 的质量分数比较差异；随后公共接口专项和全套测试均通过。未为此修改求解器或测试容差，建议将其保留为数值重复性观察项。

## 真实六文件流程

输入：本地六 STL 集成测试数据集（未纳入仓库）

新输出：本地隔离的时间戳结果目录（未纳入仓库）

- `T_CT`：warning，中置信度，参考重叠率 0.464，P90 0.3441 mm；通过原有一致性与放行门槛。warning 原因是存在分数接近但位置不同的候选。
- `T_UPPER`：success，高置信度，参考重叠率 1.000，P90 0.0000 mm。
- `T_DELTA`：success，高置信度，参考重叠率 1.000，P90 0.0000 mm。
- 三个阶段清单均由公共接口成功读取，均保留 `stage_key` 和 4×4 `transformation`。
- 五个矩阵文件与 `ct_mandible_T0.stl`、`ct_mandible_T1.stl` 均已生成。
- `T_MANDIBLE_T1 == T_DELTA @ T_CT`，最大绝对误差为 0。
- 旧项目三个阶段仍可通过公共接口读取，状态均为 success。

与 2026-09-04、核心版本 1.4.2 的旧结果比较：

| 矩阵 | 相对旋转差 | 相对平移差 |
|---|---:|---:|
| `T_CT` / `T_MANDIBLE_T0` | 0.474218° | 0.164500 mm |
| `T_UPPER` | 0.000000° | 0.000000342 mm |
| `T_DELTA` | 0.000002958° | 0.000000380 mm |
| `T_MANDIBLE_T1` | 0.474218° | 0.157882 mm |

旧结果来自 1.4.2，不是同一核心版本的接口前后对照；2.0.0 公共包装与实际求解器的同配置等价性由通用项目公共接口测试覆盖。

## 构建产物

- `dist\mandible_registration-0.1.0-py3-none-any.whl`
- SHA-256：`c7f57c23414769c2234a792ce76f9a9be44f17aeef89f3a600d06640d5e39bc7`
- 已在隔离环境中从 wheel 强制重装并通过仓库外导入、命令行 `--help` 和 `pip check`。

## GUI 验证边界

已通过自动测试覆盖 Qt 主界面状态、流程图、拖入、阶段状态、查看数据、测量点击路径，以及无界面的多区域撤销/重做、重叠拒绝、业务 profile/状态文件保存。Open3D 公开查看器类和入口可以导入。

本轮未进行人工原生交互核验，因此不能声称已重新验证真实窗口首次绘制、中文字形、鼠标套索、前表面/透选、有界组件、旋转/平移/缩放，以及保存失败时保持窗口等交互。接口指南明确允许将这些列为待人工验收项。
