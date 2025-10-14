# HDMI Development Guide

本文档说明如何在其他机器上克隆和修改 HDMI 仓库。

---

## 目录
- [首次克隆仓库](#首次克隆仓库)
- [开发环境配置](#开发环境配置)
- [日常开发流程](#日常开发流程)
- [分支管理](#分支管理)
- [常见问题](#常见问题)

---

## 首次克隆仓库

### 方法 1: HTTPS (推荐 - 更简单)

#### 步骤 1: 克隆仓库
```bash
# 克隆仓库到本地
git clone https://github.com/qqqqiiuuss/HDMI.git

# 进入项目目录
cd HDMI
```

#### 步骤 2: 配置 Git 用户信息
```bash
# 设置你的用户名和邮箱（用于 commit 记录）
git config user.name "你的用户名"
git config user.email "your.email@example.com"

# 或者设置全局配置（所有仓库都使用）
git config --global user.name "你的用户名"
git config --global user.email "your.email@example.com"
```

#### 步骤 3: 配置 Personal Access Token (PAT)

**创建 Token**：
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 勾选权限：至少选择 `repo` (完整仓库访问)
4. 点击 "Generate token"
5. **复制 token**（只显示一次！）

**配置 Token**：
```bash
# 方式 A: 使用 credential helper 存储（推荐）
git config --global credential.helper store

# 下次 push 时会提示输入：
# Username: qqqqiiuuss
# Password: <粘贴你的 Personal Access Token>
# 之后会自动记住
```

**或者使用临时缓存**（15分钟）：
```bash
git config --global credential.helper 'cache --timeout=900'
```

---

### 方法 2: SSH (适合长期使用)

#### 步骤 1: 生成 SSH Key
```bash
# 生成新的 SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# 按 Enter 使用默认路径 (~/.ssh/id_ed25519)
# 可选：设置密码保护 key

# 查看公钥
cat ~/.ssh/id_ed25519.pub
```

#### 步骤 2: 添加 SSH Key 到 GitHub
1. 复制上面命令输出的公钥
2. 访问 https://github.com/settings/keys
3. 点击 "New SSH key"
4. 标题：填写这台机器的名称（如 "Lab Server"）
5. Key：粘贴公钥
6. 点击 "Add SSH key"

#### 步骤 3: 测试连接
```bash
# 测试 SSH 连接
ssh -T git@github.com

# 应该看到：
# Hi qqqqiiuuss! You've successfully authenticated, but GitHub does not provide shell access.
```

#### 步骤 4: 克隆仓库
```bash
# 使用 SSH URL 克隆
git clone git@github.com:qqqqiiuuss/HDMI.git

# 进入项目目录
cd HDMI

# 配置用户信息
git config user.name "你的用户名"
git config user.email "your.email@example.com"
```

---

## 开发环境配置

### 1. 安装依赖

```bash
# 创建 conda 环境
conda create -n hdmi python=3.11
conda activate hdmi

# 安装 IsaacSim
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# 安装 IsaacLab（在父目录）
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i none

# 安装 HDMI
cd ../HDMI
pip install -e .
```

### 2. 准备数据（可选）

由于运动数据集太大，没有上传到 GitHub。如果需要训练，请：

**方式 A: 从原始服务器复制**
```bash
# 从原始机器复制运动数据集（如果有访问权限）
scp -r user@original-server:/path/to/twist_motion_dataset ./
scp -r user@original-server:/path/to/data/motion ./data/
```

**方式 B: 自己生成运动数据**
```bash
# 参考 GMR 子模块生成运动数据
# 详见：https://github.com/LeCAR-Lab/GMR
```

---

## 日常开发流程

### 1. 开始开发前（同步最新代码）

```bash
# 切换到 main 分支
git checkout main

# 拉取最新代码
git pull origin main
```

### 2. 创建功能分支（推荐）

```bash
# 创建并切换到新分支
git checkout -b feature/your-feature-name

# 例如：
git checkout -b feature/add-new-reward
git checkout -b fix/termination-bug
```

### 3. 修改代码并提交

```bash
# 查看修改的文件
git status

# 添加修改的文件
git add .

# 或者添加特定文件
git add active_adaptation/envs/mdp/rewards/common.py

# 提交修改
git commit -m "Add: 简短描述你的修改

- 详细描述修改内容（可选）
- 可以多行说明"

# 示例：
git commit -m "Fix: termination threshold in twist-base.yaml

- Change cum_body_pos_error threshold from 0.7 to 0.5
- Update documentation in TWIST_TO_HDMI_GUIDE.md"
```

### 4. 推送到 GitHub

```bash
# 推送当前分支到远程
git push origin feature/your-feature-name

# 如果是第一次推送这个分支
git push -u origin feature/your-feature-name
```

### 5. 创建 Pull Request（团队协作）

如果是团队协作，推荐使用 Pull Request：

1. 访问 https://github.com/qqqqiiuuss/HDMI
2. 点击 "Pull requests" → "New pull request"
3. 选择你的分支
4. 填写 PR 描述
5. 点击 "Create pull request"

### 6. 合并到主分支（个人项目可直接操作）

```bash
# 切换回 main 分支
git checkout main

# 合并功能分支
git merge feature/your-feature-name

# 推送到远程
git push origin main

# 删除本地功能分支（可选）
git branch -d feature/your-feature-name

# 删除远程功能分支（可选）
git push origin --delete feature/your-feature-name
```

---

## 快速开发流程（直接在 main 分支）

如果是个人开发，可以直接在 main 分支工作：

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 修改代码
# ... 编辑文件 ...

# 3. 查看修改
git status
git diff

# 4. 提交修改
git add .
git commit -m "Update: 描述你的修改"

# 5. 推送到 GitHub
git push origin main
```

---

## 常用 Git 命令速查

### 查看状态
```bash
# 查看当前状态
git status

# 查看修改详情
git diff

# 查看提交历史
git log --oneline --graph --all
```

### 撤销操作
```bash
# 撤销未暂存的修改（危险！）
git restore <file>
git restore .  # 撤销所有修改

# 取消暂存（但保留修改）
git restore --staged <file>

# 撤销上一次 commit（保留修改）
git reset --soft HEAD~1

# 修改上一次 commit 消息
git commit --amend -m "新的提交消息"
```

### 分支管理
```bash
# 查看所有分支
git branch -a

# 切换分支
git checkout <branch-name>

# 创建并切换到新分支
git checkout -b <new-branch-name>

# 删除本地分支
git branch -d <branch-name>

# 删除远程分支
git push origin --delete <branch-name>
```

### 同步远程
```bash
# 拉取最新代码
git pull origin main

# 或者分开操作
git fetch origin
git merge origin/main

# 强制覆盖本地代码（危险！）
git fetch origin
git reset --hard origin/main
```

---

## 分支管理策略

### 推荐分支命名规范

```
feature/<功能名>     - 新功能开发
fix/<bug名>          - Bug 修复
refactor/<模块名>    - 代码重构
docs/<文档名>        - 文档更新
test/<测试名>        - 测试相关

示例：
feature/add-twist-rewards
fix/termination-threshold
refactor/ppo-roa-modules
docs/update-installation-guide
test/add-unit-tests
```

### 工作流程示例

```bash
# 场景 1: 添加新功能
git checkout main
git pull origin main
git checkout -b feature/new-observation
# ... 开发 ...
git add .
git commit -m "Add: new observation function for proprioception"
git push origin feature/new-observation

# 场景 2: 修复 Bug
git checkout main
git pull origin main
git checkout -b fix/reward-calculation
# ... 修复 ...
git add .
git commit -m "Fix: reward calculation error in keypoint tracking"
git push origin fix/reward-calculation

# 场景 3: 代码重构
git checkout main
git pull origin main
git checkout -b refactor/clean-observations
# ... 重构 ...
git add .
git commit -m "Refactor: simplify observation function structure"
git push origin refactor/clean-observations
```

---

## 提交消息规范

### 推荐格式

```
<type>: <简短描述>

<详细描述（可选）>

<相关 Issue/PR 引用（可选）>
```

### Type 类型

- `Add`: 添加新功能
- `Fix`: 修复 Bug
- `Update`: 更新现有功能
- `Refactor`: 代码重构
- `Docs`: 文档更新
- `Test`: 测试相关
- `Chore`: 构建/工具/依赖更新
- `Style`: 代码格式调整（不影响功能）

### 示例

```bash
git commit -m "Add: TWIST teacher reward functions

- Implement keypoint position tracking with product form
- Add joint velocity tracking reward
- Update reward configuration in twist-base.yaml"

git commit -m "Fix: termination threshold too strict

The previous threshold of 0.7m was causing premature terminations.
Changed to 0.5m to allow more exploration during training.

Related to issue #12"

git commit -m "Docs: update installation guide for Isaac Lab 5.0"
```

---

## 常见问题

### Q1: 推送时提示 "Permission denied"

**解决方案**：
```bash
# 检查 remote URL
git remote -v

# 如果是 HTTPS，确保已配置 credential helper
git config --global credential.helper store

# 下次 push 时会提示输入 token
git push origin main

# 如果是 SSH，检查 SSH key
ssh -T git@github.com
```

---

### Q2: 推送时提示 "rejected: non-fast-forward"

**原因**：远程有你本地没有的提交

**解决方案**：
```bash
# 方式 A: 先拉取再推送
git pull origin main
git push origin main

# 方式 B: rebase（保持历史线性）
git pull --rebase origin main
git push origin main

# 方式 C: 强制推送（危险！会覆盖远程）
git push -f origin main
```

---

### Q3: 如何忽略本地配置文件

编辑 `.git/info/exclude`（不会被提交）：
```bash
# 编辑本地排除文件
nano .git/info/exclude

# 添加要忽略的文件
outputs/
*.log
my_local_config.yaml
```

---

### Q4: 如何撤销已推送的 commit

```bash
# 方式 A: revert（创建新 commit 撤销）
git revert <commit-hash>
git push origin main

# 方式 B: reset + 强制推送（危险！）
git reset --hard HEAD~1
git push -f origin main
```

---

### Q5: 如何解决合并冲突

```bash
# 1. 拉取时发生冲突
git pull origin main
# Auto-merging file.py
# CONFLICT (content): Merge conflict in file.py

# 2. 编辑冲突文件，找到冲突标记
# <<<<<<< HEAD
# 你的修改
# =======
# 远程的修改
# >>>>>>> origin/main

# 3. 解决冲突后
git add <conflicted-file>
git commit -m "Merge: resolve conflicts"
git push origin main
```

---

### Q6: 大文件上传失败

GitHub 限制单个文件 100MB。如果需要上传大文件：

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pth"
git lfs track "*.npz"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Add: Git LFS tracking"
git push origin main
```

---

## 多人协作最佳实践

### 1. 开发前同步
```bash
git checkout main
git pull origin main
```

### 2. 使用功能分支
```bash
git checkout -b feature/your-feature
```

### 3. 经常提交
```bash
# 小步提交，便于回滚
git commit -m "Add: reward function structure"
git commit -m "Add: reward function implementation"
git commit -m "Add: reward function tests"
```

### 4. 推送前 rebase
```bash
# 保持提交历史整洁
git checkout main
git pull origin main
git checkout feature/your-feature
git rebase main
git push origin feature/your-feature
```

### 5. Code Review
- 使用 Pull Request 进行代码审查
- 等待审查通过后再合并

---

## 有用的 Git 配置

```bash
# 彩色输出
git config --global color.ui auto

# 默认编辑器（可选）
git config --global core.editor "vim"
git config --global core.editor "nano"

# 简化命令别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.lg "log --oneline --graph --all"

# 使用别名
git st    # 等同于 git status
git co main  # 等同于 git checkout main
git lg    # 查看提交图
```

---

## 参考资源

- [GitHub Personal Access Token 文档](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [GitHub SSH 配置指南](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [Git 官方文档](https://git-scm.com/doc)
- [Pro Git 中文版](https://git-scm.com/book/zh/v2)

---

## 快速参考卡片

### 首次克隆 + 配置
```bash
git clone https://github.com/qqqqiiuuss/HDMI.git
cd HDMI
git config user.name "Your Name"
git config user.email "your.email@example.com"
git config --global credential.helper store
```

### 日常开发循环
```bash
git pull origin main          # 1. 同步
# ... 修改代码 ...            # 2. 开发
git add .                     # 3. 暂存
git commit -m "Update: ..."   # 4. 提交
git push origin main          # 5. 推送
```

### 紧急回滚
```bash
git log --oneline             # 查看历史
git reset --hard <commit-id>  # 回滚到指定提交
git push -f origin main       # 强制推送（慎用！）
```

---

**更新日期**: 2025-10-14
**维护者**: qqqqiiuuss
**问题反馈**: 在 [GitHub Issues](https://github.com/qqqqiiuuss/HDMI/issues) 提交
