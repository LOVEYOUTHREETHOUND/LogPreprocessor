import os
import jieba
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加自定义词典
def init_custom_dictionary():
    """初始化自定义词典，添加专业术语"""
    custom_words = [
        # 系统相关
        "LTWL_m", "LTWL_s", "lteenb", "gcc", "glibc", "SMP", "bandwidth", "PRACH", "DRBs", "RF0",
        
        # 网络相关
        "Cat0终端", "enb_gnb", "HO", "LTE", "重选测试", "驻网测试", "geeflex", "test", "report",
        "earfcn", "pci", "FDD", "DL", "UL", "cyclic_prefix", "prach_config_index", "prach_freq_offset",
        "delta_pucch_shift", "n_rb_cqi", "n_cs_an", "PUCCH", "ACK", "NACK", "SR", "CQI", "SRS",
        
        # 配置相关
        "global_enb_id", "cell_id", "rnti", "sfn", "channel", "n_rb_dl", "n_rb_ul", "GBR", "CFG",
        
        # 测试相关
        "驻网", "重选", "切换", "测试", "状态", "终端", "HO", "Cat0"
    ]
    for word in custom_words:
        jieba.add_word(word)

def extract_timestamp(line):
    """提取时间戳"""
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}[\s_]\d{2}[:.]\d{2}[:.]\d{2}',  # 2025-03-13 17:16:16
        r'\d{4}-\d{2}-\d{2}[\s_]\d{2}[:.]\d{2}',           # 2025-03-13 17:16
        r'Started on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})' # Started on 2025-03-13 17:16:16
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(0)
    return None

def extract_signaling(line):
    """提取信令相关内容"""
    signaling_patterns = {
        'HO': r'HO[\s_](Request|Response|Command|Complete|Failure)',
        'LTE': r'LTE[_-](重选|切换|测试|状态)',
        '终端': r'Cat0终端[_-](HO|驻网|测试)',
        '测试': r'(重选|驻网)测试',
        '配置': r'(global_enb_id|cell_id|earfcn|pci|mode|bandwidth)',
        '资源': r'(PUCCH|ACK|NACK|SR|CQI|SRS)',
        '系统': r'(LTWL_m|LTWL_s|lteenb|gcc|glibc)'
    }
    
    results = []
    for key, pattern in signaling_patterns.items():
        matches = re.finditer(pattern, line)
        for match in matches:
            results.append(match.group(0))
    return results

def is_noise_word(word):
    """判断是否为噪声词"""
    noise_words = {
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
        '或', '一个', '没有', '这个', '那个', '这样', '那样', '这些',
        '那些', '什么', '如何', '怎么', '为什么', '是否', '可以',
        '不能', '需要', '应该', '必须', '可能', '一定', '总是',
        '有时', '经常', '偶尔', '很少', '从不', '现在', '过去',
        '未来', '之前', '之后', '同时', '立即', '马上', '很快',
        '慢慢', '逐渐', '突然', '立即', '马上', '很快', '慢慢',
        '逐渐', '突然', '立即', '马上', '很快', '慢慢', '逐渐',
        '突然', '立即', '马上', '很快', '慢慢', '逐渐', '突然'
    }
    return word in noise_words or len(word) < 2 or word.isdigit()

def extract_structured_features(line):
    """提取结构化特征"""
    features = {
        'system_info': {},
        'config_params': {},
        'event_info': {},
        'error_warning': [],
        'numeric_values': []
    }
    
    # 系统信息提取
    system_patterns = {
        'version': r'version\s+([\d\.-]+)',
        'license': r'Licensed to \'([^\']+)\'',
        'global_id': r'global_enb_id=(\d+\.\d+)'
    }
    
    # 配置参数提取
    config_patterns = {
        'bandwidth': r'bandwidth=(\d+\.?\d*)',
        'earfcn': r'earfcn=(\d+)',
        'pci': r'pci=(\d+)',
        'mode': r'mode=(\w+)',
        'n_rb_dl': r'n_rb_dl=(\d+)',
        'n_rb_ul': r'n_rb_ul=(\d+)'
    }
    
    # 错误和警告信息提取
    error_patterns = [
        r'ERROR:?\s*(.*)',
        r'error:?\s*(.*)',
        r'WARNING:?\s*(.*)',
        r'warning:?\s*(.*)',
        r'FAIL:?\s*(.*)',
        r'fail:?\s*(.*)'
    ]
    
    # 提取系统信息
    for key, pattern in system_patterns.items():
        match = re.search(pattern, line)
        if match:
            features['system_info'][key] = match.group(1)
    
    # 提取配置参数
    for key, pattern in config_patterns.items():
        match = re.search(pattern, line)
        if match:
            features['config_params'][key] = float(match.group(1))
    
    # 提取错误和警告信息
    for pattern in error_patterns:
        match = re.search(pattern, line)
        if match:
            features['error_warning'].append(match.group(1))
    
    # 提取数值型信息
    numeric_matches = re.finditer(r'\b\d+\.?\d*\b', line)
    for match in numeric_matches:
        features['numeric_values'].append(float(match.group(0)))
    
    return features

def create_vector_representation(file_data):
    """创建文件的矢量表示"""
    # 1. 文本特征（使用TF-IDF）
    text_content = ' '.join(file_data['words'])
    vectorizer = TfidfVectorizer(max_features=100)
    text_vector = vectorizer.fit_transform([text_content]).toarray()
    
    # 2. 结构化特征
    structured_features = []
    for line in file_data['raw_content']:
        features = extract_structured_features(line)
        structured_features.append(features)
    
    # 3. 统计特征
    stats_features = {
        'word_count': len(file_data['words']),
        'unique_words': len(set(file_data['words'])),
        'timestamp_count': len(file_data['timestamps']),
        'signaling_count': len(file_data['signaling']),
        'error_count': sum(1 for f in structured_features if f['error_warning']),
        'numeric_count': sum(len(f['numeric_values']) for f in structured_features)
    }
    
    # 4. 时间特征
    if file_data['timestamps']:
        timestamps = pd.to_datetime(file_data['timestamps'])
        time_features = {
            'hour_mean': timestamps.dt.hour.mean(),
            'hour_std': timestamps.dt.hour.std(),
            'duration': (timestamps.max() - timestamps.min()).total_seconds()
        }
    else:
        time_features = {'hour_mean': 0, 'hour_std': 0, 'duration': 0}
    
    # 组合所有特征
    vector_rep = {
        'text_vector': text_vector,
        'structured_features': structured_features,
        'stats_features': stats_features,
        'time_features': time_features
    }
    
    return vector_rep

class LogBERT:
    def __init__(self, model_name="bert-base-chinese", num_labels=5):
        """初始化BERT模型"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True,
            output_hidden_states=True
        ).to(self.device)
        
        # 定义标签映射
        self.label_map = {
            0: "系统配置",
            1: "信令流程",
            2: "错误警告",
            3: "性能指标",
            4: "其他"
        }
        
    def preprocess_text(self, text):
        """预处理文本"""
        # 使用jieba分词
        words = jieba.cut(text, cut_all=False)
        # 过滤噪声词
        words = [w.strip() for w in words if w.strip() and not is_noise_word(w.strip())]
        # 合并为句子
        return " ".join(words)
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, log_data):
        """准备数据集"""
        texts = []
        labels = []
        
        for line in log_data['raw_content']:
            # 预处理文本
            processed_text = self.preprocess_text(line)
            if not processed_text:
                continue
                
            # 根据特征确定标签
            features = extract_structured_features(line)
            if features['system_info'] or features['config_params']:
                label = 0  # 系统配置
            elif features['error_warning']:
                label = 2  # 错误警告
            elif any(key in line for key in ['HO', 'LTE', '信令']):
                label = 1  # 信令流程
            elif any(key in line for key in ['bandwidth', 'earfcn', 'pci']):
                label = 3  # 性能指标
            else:
                label = 4  # 其他
                
            texts.append(processed_text)
            labels.append(label)
        
        # 创建数据集
        dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })
        
        # tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def train(self, log_data, output_dir="./bert_model"):
        """训练模型"""
        # 准备数据集
        dataset = self.prepare_dataset(log_data)
        
        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=500,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
        )
        
        # 定义数据整理器
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        logger.info("开始训练BERT模型...")
        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"模型已保存到: {output_dir}")
    
    def predict(self, text):
        """预测单条日志"""
        # 预处理文本
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return None
            
        # tokenize
        inputs = self.tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            
        return {
            "label": self.label_map[predicted_label],
            "probability": probabilities[0][predicted_label].item(),
            "all_probabilities": probabilities[0].tolist()
        }
    
    def analyze_log_file(self, file_data):
        """分析整个日志文件"""
        results = {
            "predictions": [],
            "label_distribution": defaultdict(int),
            "confidence_scores": []
        }
        
        for line in file_data['raw_content']:
            prediction = self.predict(line)
            if prediction:
                results["predictions"].append(prediction)
                results["label_distribution"][prediction["label"]] += 1
                results["confidence_scores"].append(prediction["probability"])
        
        # 计算统计信息
        results["total_lines"] = len(results["predictions"])
        results["avg_confidence"] = np.mean(results["confidence_scores"]) if results["confidence_scores"] else 0
        
        return results

def process_log_file(file_path):
    """处理单个日志文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        file_data = {
            'timestamps': [],
            'signaling': [],
            'words': [],
            'raw_content': content
        }
        
        for line in content:
            # 提取时间戳
            timestamp = extract_timestamp(line)
            if timestamp:
                file_data['timestamps'].append(timestamp)
            
            # 提取信令
            signaling = extract_signaling(line)
            file_data['signaling'].extend(signaling)
            
            # 分词处理
            words = jieba.cut(line, cut_all=False)
            words = [w.strip() for w in words if w.strip() and not is_noise_word(w.strip())]
            file_data['words'].extend(words)
        
        # 创建矢量表示
        file_data['vector_rep'] = create_vector_representation(file_data)
        
        # 使用BERT模型分析
        bert_model = LogBERT()
        file_data['bert_analysis'] = bert_model.analyze_log_file(file_data)
        
        return file_data
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return {'timestamps': [], 'signaling': [], 'words': [], 'raw_content': [], 'vector_rep': None, 'bert_analysis': None}

def extract_test_type(filename):
    """从文件名中提取测试类型信息"""
    # 假设文件名格式为：test_type_other_info.log
    # 例如：lte_handover_test_20240313.log
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[0]  # 返回测试类型
    return "unknown"

def group_logs_by_test_type(file_data_list):
    """按测试类型对日志文件进行分组"""
    test_type_groups = defaultdict(list)
    for file_data in file_data_list:
        test_type = extract_test_type(file_data['filename'])
        test_type_groups[test_type].append(file_data)
    return test_type_groups

def analyze_test_type_comparison(test_type_groups):
    """分析不同测试类型的对比结果"""
    comparison_results = {
        'test_type_stats': defaultdict(dict),
        'performance_comparison': defaultdict(list),
        'error_patterns': defaultdict(lambda: defaultdict(int)),
        'time_trends': defaultdict(list)
    }
    
    for test_type, files in test_type_groups.items():
        # 1. 基本统计
        comparison_results['test_type_stats'][test_type] = {
            'total_tests': len(files),
            'total_errors': sum(f['error_count'] for f in files),
            'avg_duration': np.mean([f['duration'] for f in files if 'duration' in f]),
            'success_rate': np.mean([f['success_rate'] for f in files if 'success_rate' in f])
        }
        
        # 2. 性能指标对比
        for file_data in files:
            if 'performance_metrics' in file_data:
                for metric, value in file_data['performance_metrics'].items():
                    comparison_results['performance_comparison'][f"{test_type}_{metric}"].append(value)
        
        # 3. 错误模式分析
        for file_data in files:
            if 'error_patterns' in file_data:
                for error, count in file_data['error_patterns'].items():
                    comparison_results['error_patterns'][test_type][error] += count
        
        # 4. 时间趋势分析
        for file_data in files:
            if 'timestamps' in file_data and file_data['timestamps']:
                comparison_results['time_trends'][test_type].extend(file_data['timestamps'])
    
    return comparison_results

def analyze_logs(log_dir):
    """分析日志目录中的所有日志文件"""
    print("开始分析日志文件...")
    init_custom_dictionary()
    
    all_data = {
        'file_stats': [],
        'total_timestamps': [],
        'signaling_events': defaultdict(int),
        'word_freq': Counter(),
        'vector_representations': [],
        'test_type_groups': defaultdict(list)
    }
    
    # 处理所有日志文件
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(log_dir, filename)
            print(f"正在处理文件: {filename}")
            
            file_data = process_log_file(file_path)
            if file_data:
                # 提取测试类型并分组
                test_type = extract_test_type(filename)
                all_data['test_type_groups'][test_type].append(file_data)
                
                # 更新总体统计
                all_data['total_timestamps'].extend(file_data['timestamps'])
                for sig in file_data['signaling']:
                    all_data['signaling_events'][sig] += 1
                all_data['word_freq'].update(file_data['words'])
                
                # 保存文件级别统计
                file_stats = {
                    'filename': filename,
                    'test_type': test_type,
                    'total_words': len(file_data['words']),
                    'unique_words': len(set(file_data['words'])),
                    'timestamp_count': len(file_data['timestamps']),
                    'signaling_count': len(file_data['signaling']),
                    'error_count': sum(1 for f in file_data['vector_rep']['structured_features'] if f['error_warning']) if file_data['vector_rep'] else 0
                }
                all_data['file_stats'].append(file_stats)
    
    # 生成测试类型对比分析
    comparison_results = analyze_test_type_comparison(all_data['test_type_groups'])
    
    # 生成报告
    generate_reports(log_dir, all_data)
    generate_comparison_report(os.path.join(log_dir, 'analysis_results'), comparison_results)

def generate_reports(log_dir, all_data):
    """生成分析报告"""
    # 创建输出目录
    output_dir = os.path.join(log_dir, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 基础统计报告
    df_basic = pd.DataFrame(all_data['file_stats'])
    basic_report_path = os.path.join(output_dir, 'log_analysis_basic.xlsx')
    df_basic.to_excel(basic_report_path, index=False)
    
    # 2. 信令统计报告
    df_signaling = pd.DataFrame([
        {'signaling_type': k, 'count': v}
        for k, v in all_data['signaling_events'].items()
    ])
    signaling_report_path = os.path.join(output_dir, 'log_analysis_signaling.xlsx')
    if not df_signaling.empty:
        df_signaling.to_excel(signaling_report_path, index=False)
    
    # 3. 时间分布报告
    timestamps_sorted = sorted(all_data['total_timestamps'])
    if timestamps_sorted:
        time_stats = {
            'start_time': timestamps_sorted[0],
            'end_time': timestamps_sorted[-1],
            'total_events': len(timestamps_sorted)
        }
        df_time = pd.DataFrame([time_stats])
        time_report_path = os.path.join(output_dir, 'log_analysis_time.xlsx')
        df_time.to_excel(time_report_path, index=False)
    
    # 4. 生成可视化图表
    generate_visualizations(output_dir, all_data)
    
    # 添加BERT分析报告
    if all_data['vector_representations']:
        bert_report = []
        for vec_rep in all_data['vector_representations']:
            if 'bert_analysis' in vec_rep:
                bert_report.append({
                    'filename': vec_rep['filename'],
                    'total_lines': vec_rep['bert_analysis']['total_lines'],
                    'avg_confidence': vec_rep['bert_analysis']['avg_confidence'],
                    'label_distribution': dict(vec_rep['bert_analysis']['label_distribution'])
                })
        
        df_bert = pd.DataFrame(bert_report)
        bert_report_path = os.path.join(output_dir, 'log_analysis_bert.xlsx')
        df_bert.to_excel(bert_report_path, index=False)
        print(f"5. BERT分析报告: {bert_report_path}")
    
    print("\n=== 分析报告生成完成 ===")
    print(f"1. 基础统计报告: {basic_report_path}")
    print(f"2. 信令统计报告: {signaling_report_path}")
    print(f"3. 时间分布报告: {time_report_path}")
    print(f"4. 可视化图表: {output_dir}")
    
    # 打印总体统计信息
    print("\n=== 总体统计 ===")
    print(f"处理文件总数: {len(all_data['file_stats'])}")
    print(f"识别信令类型总数: {len(all_data['signaling_events'])}")
    print(f"时间戳总数: {len(all_data['total_timestamps'])}")
    
    # 打印最常见的信令类型
    print("\n最常见的信令类型:")
    for sig_type, count in sorted(all_data['signaling_events'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{sig_type}: {count}次")
    
    # 打印最常见的词
    print("\n最常见的词:")
    for word, count in all_data['word_freq'].most_common(20):
        print(f"{word}: {count}次")

def generate_visualizations(output_dir, all_data):
    """生成可视化图表"""
    plt.style.use('seaborn')
    
    # 1. 词频统计图
    plt.figure(figsize=(15, 8))
    top_words = dict(all_data['word_freq'].most_common(20))
    plt.bar(top_words.keys(), top_words.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 词频统计')
    plt.xlabel('词语')
    plt.ylabel('出现次数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_frequency.png'))
    plt.close()
    
    # 2. 信令类型分布图
    plt.figure(figsize=(12, 6))
    signaling_data = dict(sorted(all_data['signaling_events'].items(), key=lambda x: x[1], reverse=True))
    plt.bar(signaling_data.keys(), signaling_data.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('信令类型分布')
    plt.xlabel('信令类型')
    plt.ylabel('出现次数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signaling_distribution.png'))
    plt.close()
    
    # 3. 文件大小分布图
    plt.figure(figsize=(10, 6))
    file_sizes = [stats['total_words'] for stats in all_data['file_stats']]
    plt.hist(file_sizes, bins=20)
    plt.title('日志文件大小分布')
    plt.xlabel('词数')
    plt.ylabel('文件数量')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'file_size_distribution.png'))
    plt.close()
    
    # 4. 时间分布热力图
    if all_data['total_timestamps']:
        timestamps = pd.to_datetime(all_data['total_timestamps'])
        df_time = pd.DataFrame({
            'hour': timestamps.dt.hour,
            'day': timestamps.dt.day
        })
        
        plt.figure(figsize=(12, 8))
        pivot_table = pd.crosstab(df_time['day'], df_time['hour'])
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='d')
        plt.title('日志记录时间分布热力图')
        plt.xlabel('小时')
        plt.ylabel('日期')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_distribution_heatmap.png'))
        plt.close()

def generate_comparison_report(output_dir, comparison_results):
    """生成测试类型对比分析报告"""
    # 1. 测试类型统计报告
    df_stats = pd.DataFrame(comparison_results['test_type_stats']).T
    stats_path = os.path.join(output_dir, 'test_type_comparison_stats.xlsx')
    df_stats.to_excel(stats_path)
    
    # 2. 性能对比报告
    performance_data = []
    for metric, values in comparison_results['performance_comparison'].items():
        test_type = metric.split('_')[0]
        performance_data.append({
            'test_type': test_type,
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        })
    df_performance = pd.DataFrame(performance_data)
    performance_path = os.path.join(output_dir, 'test_type_performance_comparison.xlsx')
    df_performance.to_excel(performance_path, index=False)
    
    # 3. 错误模式对比报告
    error_data = []
    for test_type, errors in comparison_results['error_patterns'].items():
        for error, count in errors.items():
            error_data.append({
                'test_type': test_type,
                'error_pattern': error,
                'count': count
            })
    df_errors = pd.DataFrame(error_data)
    errors_path = os.path.join(output_dir, 'test_type_error_patterns.xlsx')
    df_errors.to_excel(errors_path, index=False)
    
    # 4. 生成可视化图表
    # 4.1 测试类型成功率对比
    plt.figure(figsize=(12, 6))
    test_types = list(comparison_results['test_type_stats'].keys())
    success_rates = [stats['success_rate'] for stats in comparison_results['test_type_stats'].values()]
    plt.bar(test_types, success_rates)
    plt.title('Test Type Success Rate Comparison')
    plt.xlabel('Test Type')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_type_success_rate.png'))
    plt.close()
    
    # 4.2 性能指标对比箱线图
    plt.figure(figsize=(15, 8))
    performance_data = []
    labels = []
    for metric, values in comparison_results['performance_comparison'].items():
        performance_data.append(values)
        labels.append(metric)
    plt.boxplot(performance_data, labels=labels)
    plt.title('Performance Metrics Comparison by Test Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_type_performance_comparison.png'))
    plt.close()
    
    # 4.3 错误模式热力图
    error_matrix = pd.DataFrame(comparison_results['error_patterns']).fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(error_matrix, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Error Patterns by Test Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_type_error_patterns_heatmap.png'))
    plt.close()
    
    # 4.4 时间趋势图
    plt.figure(figsize=(15, 8))
    for test_type, timestamps in comparison_results['time_trends'].items():
        if timestamps:
            timestamps = pd.to_datetime(timestamps)
            plt.plot(timestamps, label=test_type)
    plt.title('Test Execution Trends by Type')
    plt.xlabel('Time')
    plt.ylabel('Test Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_type_time_trends.png'))
    plt.close()

if __name__ == "__main__":
    log_dir = "log"  # 日志文件目录
    analyze_logs(log_dir) 