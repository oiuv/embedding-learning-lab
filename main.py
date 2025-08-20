#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Learning Lab - 统一入口程序
一个友好的交互式学习平台，支持一键运行所有教程
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List
import json

# 彩色输出支持
if platform.system() == 'Windows':
    try:
        import colorama
        colorama.init()
    except ImportError:
        pass

# 颜色定义
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title: str):
    """打印标题"""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{title:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_menu(options: Dict[str, str], title: str = "选择菜单"):
    """打印菜单"""
    print(f"\n{Colors.OKBLUE}{title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'='*40}{Colors.ENDC}")
    for key, value in options.items():
        print(f"{Colors.OKGREEN}{key}{Colors.ENDC}. {value}")

def get_choice(max_choice: int) -> int:
    """获取用户选择"""
    while True:
        try:
            choice = input(f"\n{Colors.WARNING}请输入选项 (1-{max_choice}) 或 q 退出: {Colors.ENDC}").strip()
            if choice.lower() == 'q':
                return -1
            choice = int(choice)
            if 1 <= choice <= max_choice:
                return choice
            else:
                print(f"{Colors.FAIL}请输入 1-{max_choice} 之间的数字{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.FAIL}请输入有效数字{Colors.ENDC}")

def run_script(script_path: str):
    """运行Python脚本"""
    try:
        print(f"\n{Colors.OKBLUE}正在运行: {script_path}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'-'*50}{Colors.ENDC}")
        
        # 检查文件是否存在
        if not os.path.exists(script_path):
            print(f"{Colors.FAIL}文件不存在: {script_path}{Colors.ENDC}")
            return
            
        # 运行脚本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n{Colors.OKGREEN}✓ 运行成功!{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}✗ 运行失败，返回码: {result.returncode}{Colors.ENDC}")
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}用户中断运行{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}运行出错: {e}{Colors.ENDC}")

def check_requirements():
    """检查环境要求"""
    print(f"{Colors.OKBLUE}检查环境...{Colors.ENDC}")
    
    # 检查API密钥
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print(f"{Colors.WARNING}⚠  未找到 DASHSCOPE_API_KEY 环境变量{Colors.ENDC}")
        print(f"{Colors.WARNING}   请设置: export DASHSCOPE_API_KEY=\"your-key\"{Colors.ENDC}")
        return False
    else:
        print(f"{Colors.OKGREEN}✓ API密钥已配置{Colors.ENDC}")
    
    return True

def get_tutorial_list() -> Dict[str, List[Dict]]:
    """获取教程列表"""
    return {
        "基础概念 (01-basics)": [
            {"name": "什么是文本嵌入", "file": "01-basics/01-what-is-embedding.py"},
            {"name": "获取第一个向量", "file": "01-basics/02-first-embedding.py"},
            {"name": "相似度计算", "file": "01-basics/03-similarity-calculation.py"},
            {"name": "向量操作", "file": "01-basics/04-vector-operations.py"}
        ],
        "中级应用 (02-intermediate)": [
            {"name": "语义搜索系统", "file": "02-intermediate/01-semantic-search.py"},
            {"name": "文本分类", "file": "02-intermediate/02-text-classification.py"},
            {"name": "推荐系统", "file": "02-intermediate/03-text-recommendation.py"},
            {"name": "聚类分析", "file": "02-intermediate/04-clustering-analysis.py"}
        ],
        "高级功能 (03-advanced)": [
            {"name": "智能知识库", "file": "03-advanced/01-knowledge-base.py"},
            {"name": "异常检测", "file": "03-advanced/02-anomaly-detection.py"},
            {"name": "数据可视化", "file": "03-advanced/03-visualization.py"},
            {"name": "性能优化", "file": "03-advanced/04-performance-optimization.py"}
        ],
        "文本排序 (03-text-reranking)": [
            {"name": "排序模型基础", "file": "03-text-reranking/01-text-rerank-basics.py"},
            {"name": "系统集成", "file": "03-text-reranking/02-integration-guide.py"},
            {"name": "对比演示", "file": "03-text-reranking/03-comparison-demo.py"},
            {"name": "性能基准", "file": "03-text-reranking/04-performance-benchmark.py"}
        ],
        "实战项目 (04-projects)": [
            {"name": "智能问答系统", "file": "04-projects/01-smart-qa-system/main.py"},
            {"name": "内容推荐引擎", "file": "04-projects/02-content-recommendation/main.py"},
            {"name": "语义搜索引擎", "file": "04-projects/03-semantic-search-engine/main.py"},
            {"name": "文档分析工具", "file": "04-projects/04-document-analyzer/main.py"}
        ]
    }

def show_learning_path():
    """显示学习路径"""
    print_header("📚 推荐学习路径")
    
    paths = {
        "1": {
            "name": "零基础入门路径",
            "steps": [
                "01-basics → 理解基础概念",
                "02-intermediate/01 → 体验语义搜索",
                "03-text-reranking/01 → 了解排序优化"
            ]
        },
        "2": {
            "name": "业务应用路径",
            "steps": [
                "02-intermediate → 掌握4大应用场景",
                "03-advanced → 学习高级优化",
                "04-projects → 完成实战项目"
            ]
        },
        "3": {
            "name": "完整精通路径",
            "steps": [
                "全部基础教程 → 深入理解原理",
                "全部中级教程 → 掌握核心应用",
                "全部高级教程 → 达到专家水平"
            ]
        }
    }
    
    for key, path in paths.items():
        print(f"\n{Colors.OKGREEN}{key}. {path['name']}{Colors.ENDC}")
        for i, step in enumerate(path['steps'], 1):
            print(f"   {i}. {step}")

def interactive_mode():
    """交互式模式"""
    tutorials = get_tutorial_list()
    
    while True:
        print_header("🚀 Embedding Learning Lab - 交互式学习平台")
        
        # 显示主菜单
        menu_options = {
            "1": "按阶段学习",
            "2": "按功能体验",
            "3": "推荐学习路径",
            "4": "一键运行所有基础",
            "5": "检查环境",
            "q": "退出"
        }
        
        print_menu(menu_options, "主菜单")
        
        choice = get_choice(5)
        if choice == -1:
            print(f"\n{Colors.OKGREEN}感谢使用！再见！{Colors.ENDC}")
            break
        
        if choice == 1:  # 按阶段学习
            stage_menu(tutorials)
        elif choice == 2:  # 按功能体验
            function_menu(tutorials)
        elif choice == 3:  # 推荐学习路径
            show_learning_path()
            input("\n按回车键继续...")
        elif choice == 4:  # 一键运行所有基础
            run_all_basic(tutorials)
        elif choice == 5:  # 检查环境
            check_requirements()
            input("\n按回车键继续...")

def stage_menu(tutorials):
    """阶段选择菜单"""
    while True:
        print_header("📖 按阶段学习")
        
        stages = list(tutorials.keys())
        menu = {str(i+1): stage for i, stage in enumerate(stages)}
        menu[str(len(stages)+1)] = "返回主菜单"
        
        print_menu(menu, "学习阶段")
        
        choice = get_choice(len(stages) + 1)
        if choice == -1 or choice == len(stages) + 1:
            break
        
        stage_name = stages[choice-1]
        tutorial_menu(tutorials[stage_name], stage_name)

def tutorial_menu(tutorials, category_name):
    """教程选择菜单"""
    while True:
        print_header(f"📚 {category_name}")
        
        menu = {str(i+1): t["name"] for i, t in enumerate(tutorials)}
        menu[str(len(tutorials)+1)] = "返回上一级"
        
        print_menu(menu, "选择教程")
        
        choice = get_choice(len(tutorials) + 1)
        if choice == -1 or choice == len(tutorials) + 1:
            break
        
        tutorial = tutorials[choice-1]
        run_script(tutorial["file"])
        
        input("\n按回车键继续...")

def function_menu(tutorials):
    """功能体验菜单"""
    functions = {
        "1": {"name": "语义搜索体验", "files": [
            "01-basics/03-similarity-calculation.py",
            "02-intermediate/01-semantic-search.py"
        ]},
        "2": {"name": "智能推荐体验", "files": [
            "02-intermediate/02-text-classification.py",
            "02-intermediate/03-text-recommendation.py"
        ]},
        "3": {"name": "文本分类体验", "files": [
            "02-intermediate/02-text-classification.py"
        ]},
        "4": {"name": "高级排序体验", "files": [
            "03-text-reranking/01-text-rerank-basics.py",
            "03-text-reranking/02-integration-guide.py"
        ]},
        "5": {"name": "完整项目体验", "files": [
            "04-projects/01-smart-qa-system/main.py"
        ]},
        "6": {"name": "返回主菜单", "files": []}
    }
    
    while True:
        print_header("🎯 按功能体验")
        print_menu({k: v["name"] for k, v in functions.items()}, "选择功能")
        
        choice = get_choice(len(functions))
        if choice == -1 or choice == len(functions):
            break
        
        func = functions[str(choice)]
        for file in func["files"]:
            run_script(file)

def run_all_basic(tutorials):
    """一键运行所有基础教程"""
    print_header("🚀 一键运行所有基础教程")
    
    basic_tutorials = tutorials["基础概念 (01-basics)"]
    
    for tutorial in basic_tutorials:
        print(f"\n{Colors.OKBLUE}即将运行: {tutorial['name']}{Colors.ENDC}")
        if input("运行？[Y/n]: ").lower() != 'n':
            run_script(tutorial["file"])
        else:
            break

def quick_start():
    """快速启动模式"""
    print_header("🚀 快速体验模式")
    
    if not check_requirements():
        return
    
    print(f"{Colors.OKGREEN}✓ 环境检查完成，准备快速体验...{Colors.ENDC}")
    
    # 运行基础示例
    basic_scripts = [
        "01-basics/01-what-is-embedding.py",
        "01-basics/02-first-embedding.py",
        "02-intermediate/01-semantic-search.py"
    ]
    
    for script in basic_scripts:
        if os.path.exists(script):
            run_script(script)
            if input("\n继续下一个？[Y/n]: ").lower() == 'n':
                break

def main():
    """主函数"""
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "quick":
            quick_start()
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.OKGREEN}感谢使用！再见！{Colors.ENDC}")

if __name__ == "__main__":
    main()