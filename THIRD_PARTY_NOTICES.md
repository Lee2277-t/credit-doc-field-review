# 第三方组件与许可（Third-Party Notices）

本项目以 **GNU Affero General Public License v3.0 only（AGPL-3.0-only）** 发布，全文见 [LICENSE](LICENSE)。
之所以选 AGPL-3.0，是因为核心 PDF 解析直接依赖 PyMuPDF，它只有 AGPL-3.0 与 Artifex 商业授权两条路。

```
提取文字工具（离线信贷资料结构化引擎）
Copyright (C) 2026 提取文字工具 contributors

本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共许可证
第 3 版的条款重新分发和/或修改它。
本程序按「无任何担保」分发，甚至不包含适销性或特定用途适用性的默示担保。
详见 LICENSE 文件。
```

## 直接依赖（版本为本仓库测试所用版本）

| 组件 | 版本 | 许可证 | 说明 |
|---|---|---|---|
| PyMuPDF | 2.0 | AGPL-3.0 或 Artifex 商业授权（双许可） | PDF 渲染、文本/坐标提取。**决定了本项目的许可证**。不适用 AGPL 的场景需向 Artifex 购买商业授权后替换。 |
| pdfplumber | 0.11.10 | MIT | 表格线框与单元格结构（第二条表格通路）。 |
| rapidocr-onnxruntime | 1.4.4 | Apache-2.0 | 扫描件 OCR。包内自带 PP-OCR 系列 ONNX 模型，随包以 Apache-2.0 分发。 |
| onnxruntime | 1.31.0 | MIT | OCR 推理运行时。 |
| opencv-python | 5.0.0.93 | Apache-2.0 | 图像预处理（去倾斜、透视矫正、去反光、CLAHE）。 |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 数值计算。 |
| PyYAML | 6.0.3 | MIT | 读取 `config/scenario.yaml`。 |
| scikit-learn | 1.9.1 | BSD-3-Clause | 置信度校准（logistic + isotonic）。 |
| Pillow | 12.2.0 | MIT-CMU | 图片解码与结果页素材处理。 |

安装：`pip install -r requirements.txt`（已锁定到上表版本，Python 3.11.9 实测）。
各许可证全文随对应发行包分发，可在 `pip show -f <包名>` 指向的目录中找到。

## 关于 PyMuPDF 的特别提示

- 本仓库整体按 AGPL-3.0 发布，因此**通过网络提供本工具的服务时，也必须按 AGPL-3.0 提供完整源码**（AGPL 第 13 条）。
- 若你需要闭源分发、或所在组织禁止 AGPL，请向 Artifex 购买商业授权（<https://artifex.com/licensing/>），本项目的其余代码不受影响。

## 关于合成语料与模型产物

- 公开演示使用生成脚本创建的虚构样本与标注。
- 本地输入、解析缓存、评估与校准产物可能包含从使用者资料派生的数据，均由 Git 忽略（见 `.gitignore`）。

## 借鉴说明

设计思路参考过若干开源工具与业界资料的公开说法（版面分块、跨页表格续接、校准与验收分层），
但**未复制粘贴任何第三方项目的源代码或文档正文**；全部实现、测试与文档均为本项目自行编写。
