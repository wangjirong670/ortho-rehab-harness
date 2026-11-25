# 私有防泄漏评测集目录（Private Test Cases）

本目录用于存放**不对外公开**的高价值红队陷阱用例，以防测试题进入主流大模型训练语料导致评测失真。

- `.gitignore` 已自动忽略本目录下的 `*.yaml`、`*.yml`、`*.json` 文件。
- 私有病例文件格式与 `../public/` 完全一致，运行 `python scripts/validate_kb.py` 时会自动一并校验。
