#!/usr/bin/env python3
"""
API Key 检查工具
用于验证 HippoRAG 所需的 API keys 是否正确设置
"""

import os
import argparse
from typing import Dict, List

def check_api_key_format(key: str, key_type: str) -> bool:
    """检查 API key 格式是否正确"""
    if key_type == 'openai':
        return key.startswith('sk-') and len(key) > 20
    elif key_type == 'azure':
        return len(key) > 10  # Azure key 格式比较灵活
    elif key_type == 'huggingface':
        return key.startswith('hf_') and len(key) > 20
    return len(key) > 5  # 通用检查

def mask_api_key(key: str) -> str:
    """遮蔽 API key 的敏感部分"""
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"

def check_environment_variables() -> Dict[str, any]:
    """检查环境变量中的 API keys"""
    keys_to_check = {
        'OPENAI_API_KEY': 'openai',
        'AZURE_OPENAI_API_KEY': 'azure', 
        'HF_TOKEN': 'huggingface',
        'ANTHROPIC_API_KEY': 'anthropic'
    }
    
    results = {}
    
    print("🔍 检查环境变量中的 API Keys...")
    print("-" * 50)
    
    for env_var, key_type in keys_to_check.items():
        key_value = os.getenv(env_var)
        
        if key_value:
            is_valid = check_api_key_format(key_value, key_type)
            masked_key = mask_api_key(key_value)
            
            if is_valid:
                print(f"✅ {env_var}: {masked_key} (格式正确)")
                results[env_var] = {'status': 'valid', 'value': key_value}
            else:
                print(f"⚠️  {env_var}: {masked_key} (格式可能不正确)")
                results[env_var] = {'status': 'invalid', 'value': key_value}
        else:
            print(f"❌ {env_var}: 未设置")
            results[env_var] = {'status': 'missing', 'value': None}
    
    return results

def test_openai_connection(api_key: str) -> bool:
    """测试 OpenAI API 连接"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # 尝试一个简单的请求
        response = client.models.list()
        return True
    except Exception as e:
        print(f"❌ OpenAI API 连接测试失败: {str(e)}")
        return False

def provide_setup_instructions():
    """提供设置说明"""
    print("\n📝 API Key 设置说明:")
    print("-" * 50)
    
    print("\n🔑 OpenAI API Key:")
    print("1. 访问 https://platform.openai.com/")
    print("2. 登录并进入 API Keys 页面")
    print("3. 创建新的 secret key")
    print("4. 设置环境变量:")
    print("   Linux/Mac: export OPENAI_API_KEY='sk-your-key-here'")
    print("   Windows:   set OPENAI_API_KEY=sk-your-key-here")
    
    print("\n🔑 Azure OpenAI API Key:")
    print("1. 登录 Azure Portal")
    print("2. 找到您的 OpenAI 资源")
    print("3. 在 'Keys and Endpoint' 页面获取 API key")
    print("4. 设置环境变量:")
    print("   Linux/Mac: export AZURE_OPENAI_API_KEY='your-azure-key'")
    print("   Windows:   set AZURE_OPENAI_API_KEY=your-azure-key")
    
    print("\n🔑 Hugging Face Token:")
    print("1. 访问 https://huggingface.co/settings/tokens")
    print("2. 创建新的 token")
    print("3. 设置环境变量:")
    print("   Linux/Mac: export HF_TOKEN='hf_your-token-here'")
    print("   Windows:   set HF_TOKEN=hf_your-token-here")

def main():
    parser = argparse.ArgumentParser(description="检查 HippoRAG API Keys 设置")
    parser.add_argument('--test-connection', action='store_true',
                       help='测试 API 连接（需要安装相应的库）')
    parser.add_argument('--show-instructions', action='store_true',
                       help='显示 API key 设置说明')
    
    args = parser.parse_args()
    
    print("🚀 HippoRAG API Key 检查工具")
    print("=" * 50)
    
    # 检查环境变量
    results = check_environment_variables()
    
    # 统计结果
    valid_keys = sum(1 for r in results.values() if r['status'] == 'valid')
    invalid_keys = sum(1 for r in results.values() if r['status'] == 'invalid')
    missing_keys = sum(1 for r in results.values() if r['status'] == 'missing')
    
    print(f"\n📊 检查结果统计:")
    print(f"✅ 有效的 API Keys: {valid_keys}")
    print(f"⚠️  格式可能不正确: {invalid_keys}")
    print(f"❌ 未设置的 API Keys: {missing_keys}")
    
    # 测试连接（如果请求）
    if args.test_connection:
        print(f"\n🔗 测试 API 连接...")
        print("-" * 30)
        
        if results.get('OPENAI_API_KEY', {}).get('status') == 'valid':
            print("🧪 测试 OpenAI API 连接...")
            if test_openai_connection(results['OPENAI_API_KEY']['value']):
                print("✅ OpenAI API 连接成功")
            else:
                print("❌ OpenAI API 连接失败")
        else:
            print("⚠️  跳过 OpenAI API 连接测试（API key 无效或未设置）")
    
    # 显示设置说明（如果请求）
    if args.show_instructions or missing_keys > 0:
        provide_setup_instructions()
    
    # 针对 HippoRAG 的特定建议
    print(f"\n💡 针对 HippoRAG 的建议:")
    print("-" * 30)
    
    if results.get('OPENAI_API_KEY', {}).get('status') == 'valid':
        print("✅ 可以使用 OpenAI 模型 (gpt-4o-mini, gpt-4 等)")
    else:
        print("⚠️  无法使用 OpenAI 模型，建议:")
        print("   1. 设置 OPENAI_API_KEY 使用 OpenAI 模型")
        print("   2. 或使用本地部署的模型（vLLM, Ollama 等）")
    
    if results.get('HF_TOKEN', {}).get('status') == 'valid':
        print("✅ 可以下载 Hugging Face 模型")
    else:
        print("⚠️  建议设置 HF_TOKEN 以便下载受限模型")
    
    print(f"\n🎯 快速开始:")
    print("1. 设置必要的 API keys")
    print("2. 运行: python export_knowledge_graph.py --dataset sample")
    print("3. 或运行: python demo_save_kg.py")

if __name__ == "__main__":
    main()