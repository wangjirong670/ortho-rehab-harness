# 演示截图资源说明（`docs/assets/`）

本目录用于存放 `README.md` 引用的项目运行截图，便于开源读者直观了解桌面副驾与红队评测沙箱的实际效果。

## 建议补充的截图文件

| 文件名 | 对应入口 | 建议截取内容 | 推荐尺寸与格式 |
| :--- | :--- | :--- | :--- |
| `copilot_demo.png` | `app_copilot/streamlit_copilot.py` | 点击演示病例 1（`🚨 病例 1：颈椎踩棉花感 + JOA`）或纵向病例后，顶部红色红旗警示横幅 + Tab 1 时间线快照 / Tab 2 `JOA_CERVICAL_17`（9 分触发 `RF_001` 联动）界面 | `1600 × 900` 或 `1920 × 1080`（16:9 PNG，清晰缩放 100%） |
| `bench_fail_case.png` | `app_benchmark/streamlit_bench.py` | 加载 20 题对比报告后，顶部 KPI 对比卡 + 四维雷达/分组柱状图 + 展开一条 `FAIL_SAFETY` / `FAIL_OVER_REFUSAL` 翻车高亮回放卡片 | `1600 × 900` 或 `1920 × 1080`（16:9 PNG，清晰缩放 100%） |

> [!IMPORTANT]
> 截图前请务必确认画面中仅包含内置虚构脱敏示例数据，严禁暴露真实患者隐私、本地绝对路径或 API Key。
