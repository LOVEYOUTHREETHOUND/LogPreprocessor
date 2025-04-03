# Log Processor

一个用于分析和处理特定格式日志文件的 Python 工具，主要用于分析网络测试相关的日志数据，包括信令、时间戳和关键词的提取与统计。

## 功能特点

- 支持多种日志文件格式的处理
- 自动提取和分析时间戳信息
- 识别和统计特定信令事件（如 HO、LTE 相关事件等）
- 中文分词处理和关键词提取
- 生成多种格式的分析报告（Excel）

## 环境要求

- Python 3.x
- 依赖包：
  - jieba==0.42.1 (中文分词)
  - pandas==2.2.1 (数据处理与报表生成)

## 安装说明

1. 克隆项目到本地
2. 创建并激活虚拟环境（推荐）
3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 将需要分析的日志文件放入 `log` 目录
2. 运行分析脚本：
```bash
python log_analyzer.py
```

## 输出说明

程序会在 log 目录下生成三种分析报告：

1. `log_analysis_basic.xlsx`: 基础统计报告
   - 文件名
   - 总词数
   - 独立词数
   - 时间戳数量
   - 信令数量
   - 常见词统计

2. `log_analysis_signaling.xlsx`: 信令统计报告
   - 信令类型
   - 出现次数

3. `log_analysis_time.xlsx`: 时间分布报告
   - 开始时间
   - 结束时间
   - 总事件数

## 支持的日志类型

- geeflex 测试日志
- enb_gnb 相关日志
- Cat0终端测试日志
- LTE相关测试日志

## 自定义配置

程序支持自定义词典配置，可以在 `init_custom_dictionary()` 函数中添加专业术语。

## 注意事项

- 确保日志文件使用 UTF-8 编码
- 大文件处理可能需要较长时间
- Excel 报告会覆盖同名文件

## 目录结构

```
LogProcessor/
├── log/                    # 日志文件目录
├── log_analyzer.py         # 主程序
├── requirements.txt        # 依赖配置
└── README.md              # 说明文档
``` 