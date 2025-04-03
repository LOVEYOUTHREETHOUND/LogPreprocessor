import os
import jieba
import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime

# 添加自定义词典
def init_custom_dictionary():
    """初始化自定义词典，添加专业术语"""
    custom_words = [
        "LTWL_m",
        "LTWL_s",
        "Cat0终端",
        "enb_gnb",
        "HO",
        "LTE",
        "重选测试",
        "驻网测试",
        "geeflex",
        "test",
        "report"
    ]
    for word in custom_words:
        jieba.add_word(word)

def extract_timestamp(line):
    """提取时间戳"""
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}[\s_]\d{2}[:.]\d{2}[:.]\d{2}',
        r'\d{4}-\d{2}-\d{2}[\s_]\d{2}[:.]\d{2}'
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
        '测试': r'(重选|驻网)测试'
    }
    
    results = []
    for key, pattern in signaling_patterns.items():
        matches = re.finditer(pattern, line)
        for match in matches:
            results.append(match.group(0))
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
            words = [w.strip() for w in words if w.strip() and len(w.strip()) > 1]
            file_data['words'].extend(words)
        
        return file_data
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return {'timestamps': [], 'signaling': [], 'words': [], 'raw_content': []}

def analyze_logs(log_dir):
    """分析日志目录中的所有日志文件"""
    print("开始分析日志文件...")
    init_custom_dictionary()
    
    all_data = {
        'file_stats': [],
        'total_timestamps': [],
        'signaling_events': defaultdict(int),
        'word_freq': Counter()
    }
    
    # 处理所有日志文件
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(log_dir, filename)
            print(f"正在处理文件: {filename}")
            
            file_data = process_log_file(file_path)
            
            # 更新总体统计
            all_data['total_timestamps'].extend(file_data['timestamps'])
            for sig in file_data['signaling']:
                all_data['signaling_events'][sig] += 1
            all_data['word_freq'].update(file_data['words'])
            
            # 保存文件级别统计
            file_stats = {
                'filename': filename,
                'total_words': len(file_data['words']),
                'unique_words': len(set(file_data['words'])),
                'timestamp_count': len(file_data['timestamps']),
                'signaling_count': len(file_data['signaling']),
                'top_words': dict(Counter(file_data['words']).most_common(10)),
                'timestamps': file_data['timestamps'][:5],
                'signaling': file_data['signaling'][:5]
            }
            all_data['file_stats'].append(file_stats)
    
    # 生成报告
    generate_reports(log_dir, all_data)

def generate_reports(log_dir, all_data):
    """生成分析报告"""
    # 1. 基础统计报告
    df_basic = pd.DataFrame(all_data['file_stats'])
    basic_report_path = os.path.join(log_dir, 'log_analysis_basic.xlsx')
    df_basic.to_excel(basic_report_path, index=False)
    
    # 2. 信令统计报告
    df_signaling = pd.DataFrame([
        {'signaling_type': k, 'count': v}
        for k, v in all_data['signaling_events'].items()
    ])
    signaling_report_path = os.path.join(log_dir, 'log_analysis_signaling.xlsx')
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
        time_report_path = os.path.join(log_dir, 'log_analysis_time.xlsx')
        df_time.to_excel(time_report_path, index=False)
    
    print("\n=== 分析报告生成完成 ===")
    print(f"1. 基础统计报告: {basic_report_path}")
    print(f"2. 信令统计报告: {signaling_report_path}")
    print(f"3. 时间分布报告: {time_report_path}")
    
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

if __name__ == "__main__":
    log_dir = "log"  # 日志文件目录
    analyze_logs(log_dir) 