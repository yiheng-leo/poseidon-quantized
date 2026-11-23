# 连接 Isambard

从本机先 SSH 到 UCL 的 `canada`，再从 `canada` 连 Isambard。两个 SSH 命令分别在对应机器上执行。

```text
本机 → canada → u6abp.aip2.isambard → cjhome/poseidon-quantized
```

## 交互连接

```bash
# 本机
ssh canada

# canada 上
cd MyHome
ssh u6abp.aip2.isambard

# Isambard 登录节点上
cd cjhome/poseidon-quantized
hostname
pwd -P
git branch --show-current
```

项目物理路径为 `/lus/lfs1aip2/projects/u6abp/ritchie_workspace/poseidon-quantized`，Git 分支为 `ritchie`。`hostname` 可能返回不同的 `login*` 名称。连接失败时先检查哪一跳失败，沿用已配置的 SSH 认证及主机指纹验证。

## Agent 非交互连接

通过两层 stdin 脚本指定 Bash，可避免 `canada` 的默认登录 shell 对引号和变量进行二次解析。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 canada 'bash -s' <<'UCL_BRIDGE'
set -e
cd MyHome
ssh -o BatchMode=yes -o ConnectTimeout=15 u6abp.aip2.isambard 'bash -s' <<'ISAMBARD'
set -e
cd cjhome/poseidon-quantized
hostname
pwd -P
git status --short
ISAMBARD
UCL_BRIDGE
```

只检查 Git、Slurm、日志或 JSON 时可以直接执行命令。需要仓库的 Python 依赖时在 Isambard 激活现有环境。

```bash
source "$HOME/miniconda3/bin/activate" quant
```

训练和评估通过 `sbatch` 提交，启动模板在 `launch_scripts/poseidon/`。具体进度、结果和恢复核对流程见 [IsambardAgent.md](IsambardAgent.md)。
