#!/bin/bash

# HippoRAG 离线运行脚本
# 解决本地模型加载时的网络连接问题

echo "🚀 HippoRAG 离线运行脚本"
echo "=========================="

# 设置离线环境变量
echo "🔧 设置离线环境变量..."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "✅ 已设置离线环境变量:"
echo "   HF_HUB_OFFLINE=1"
echo "   TRANSFORMERS_OFFLINE=1"
echo "   HF_HUB_DISABLE_TELEMETRY=1"
echo "   HF_HUB_DISABLE_PROGRESS_BARS=1"

# 设置本地模型路径
LOCAL_MODEL_PATH="/mnt/data/hy/GPT/models/nvidia/NV-Embed-v2"

echo ""
echo "📁 使用本地模型路径: $LOCAL_MODEL_PATH"

# 检查模型文件是否存在
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $LOCAL_MODEL_PATH"
    echo "请检查路径是否正确"
    exit 1
fi

# 检查关键文件
echo ""
echo "🔍 检查模型文件..."
required_files=("config.json" "tokenizer_config.json" "tokenizer.json")

for file in "${required_files[@]}"; do
    if [ -f "$LOCAL_MODEL_PATH/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
    fi
done

# 运行示例
echo ""
echo "🎯 运行离线演示..."

# 方法1: 运行知识图谱导出
echo "1. 运行知识图谱导出 (离线模式)..."
python export_knowledge_graph.py \
    --dataset sample \
    --embedding_path "$LOCAL_MODEL_PATH" \
    --offline \
    --export_format json

echo ""
echo "2. 或者运行基本演示 (离线模式)..."
python demo_save_kg.py \
    --embedding_path "$LOCAL_MODEL_PATH" \
    --offline

echo ""
echo "✅ 离线运行完成!"
echo ""
echo "💡 如果仍有网络连接问题，请确保:"
echo "   1. 模型文件完整"
echo "   2. 使用 --offline 参数"
echo "   3. 或手动设置环境变量:"
echo "      export HF_HUB_OFFLINE=1"
echo "      export TRANSFORMERS_OFFLINE=1"