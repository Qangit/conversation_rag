import os
# 强制指定国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"                # 禁用XET
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
from huggingface_hub import snapshot_download

repo_id = "BAAI/bge-m3" # 建议选择这个稳定的版本
local_dir = "/root/bge-m3"

print(f"开始从镜像站下载 {repo_id}...")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token="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", # 填入你的 Hugging Face Token
    endpoint="https://hf-mirror.com",
    resume_download=True, # 开启断点续传
    max_workers=4        # 适当的并发数
)

print("下载完成！")