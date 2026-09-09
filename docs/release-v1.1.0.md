# v1.1.0 Windows 打包验证

时间：2026-09-10T00:34:00+08:00。

## 交付物

- `MandibleRegistration-v1.1.0-win64.zip`：213,482,423 字节（约 203.6 MiB）。
- 解压后共 838 个文件、643,109,289 字节；入口为 `MandibleRegistration-v1.1.0.exe`，文件版本和产品版本均为 1.1.0。
- 附带使用说明、BSD-3-Clause 项目许可证、第三方声明、4 份 Qt/PySide6/Shiboken 许可证和 ZIP 的 SHA-256 校验文件。
- ZIP 不含 STL、医学影像、项目结果或开发虚拟环境。完整文件夹运行，不可只复制 EXE。

ZIP SHA-256：

```text
a9b0fdda73c3be8c1960a4db82ff11c7d40d21c4e931fa65138080d3b0cf93d6
```

EXE SHA-256：

```text
e7d903c958e6399e9a8c7c9598183ed06f11cae97234179bb7353679860412b6
```

## 打包修正

PyInstaller 原配置从环境中已安装的包收集 SVG，可能遗漏当前源码中新加的流程图。现在应用资源和模块均直接从当前 checkout 收集，通用配准核心继续使用已安装的正式依赖。打包脚本逐项校验当前 SVG、图标与冻结包内文件的 SHA-256，缺失或版本不一致即停止生成 ZIP。

本次已确认新版流程 SVG 与源码逐字节一致，冻结模块包含 `section_geometry`、`section_viewer`、`view_interaction`。使用说明同步改为剖面测量、双窗口和当前鼠标操作。

## 验证

- 最终测试：**139 passed，17.59 s**，含 3 项资源遗漏/陈旧版本防回归检查。
- 最终 ZIP CRC 全量检查、SHA-256、解压和资源/模块清单验证通过。
- 从独立解压目录启动实际 EXE，主界面版本、新版流程图和导入入口显示正常。
- 从源码目录之外启动实际 EXE 的六 STL 无界面配准，退出码 **0**，耗时 **272.94 s**。生成 schema 3 项目、10 项输出；输出哈希全部一致，`T_MANDIBLE_T1 = T_DELTA @ T_CT` 校验通过，无 `failure.json`。
- 该数据的 T_CT 一致性门槛通过，阶段报告仍保留 `warning` / 中可信度，P90 为 0.3441 mm；T_UPPER、T_DELTA 为 `success` / 高可信度。完成运行不等同于验证临床准确性。
- 当前源码的双侧剖面交互、测距导出和独立原生窗口验证见 [交互修订记录](condyle_sections_revision.md)。最终冻结包已验证主界面及完整配准，未重复完成剖面取点的人工交互验收。

构建使用 Python 3.12.10、PyInstaller 6.22.2 和项目锁定依赖。机器上的本地验证日志、解压副本与结果保存在父工作区的 `analysis_output/release_v1.1.0_20260910/`，不提交病例或生成结果。
