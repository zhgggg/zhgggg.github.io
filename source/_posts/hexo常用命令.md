---
title: Hexo常用命令
date: 2025-07-29 20:17:53
categories:
  - 折腾点啥
tags:
  - Hexo
  - 运维
  - Linux
---

# 添加所有更改（包括新文章）
```bash
git add .
```
# 提交更改
```bash
git commit -m "添加新文章：我的新文章标题"
```
# 推送到GitHub仓库
```bash
git push origin main
```
# 快速创建草稿

```bash
hexo new draft "未完成的文章"
```
文件会生成在 source/_drafts/，不会发布到线上，完成后执行：

```bash
hexo publish "未完成的文章"
```
# 添加本地图片

将图片放入 source/images/ 目录

在文章中引用：

```markdown
![图片描述](/images/your-image.jpg)
```
# 多设备同步写作

```bash
# 在其他设备克隆仓库后
git clone https://github.com/你的用户名/仓库.git
cd 仓库
npm install  # 安装Hexo依赖
```