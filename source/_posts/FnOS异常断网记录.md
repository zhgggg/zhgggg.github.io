---
title: FnOS异常断网记录
date: 2025-07-31 05:51:21
categories:
  - 折腾点啥
tags:
  - NAS
  - FnOS
  - Linux
---
# 前言
昨日备份本地文件时突然掉线，此前也有过类似情况，但是路由器手动指配ip地址后不再出现，查询社区帖子发现这种情况较为普遍，参考了别人提出的[解决方案](https://club.fnnas.com/forum.php?mod=viewthread&tid=28479),记录一下


# 死机情况描述
1. 无网络链接
2. 各指示灯正常亮起

# 解决方案
连接ssh，关闭swap。命令如下：
~~~~bash
swapoff -a #临时关闭swap
swapon -a #启用swap
~~~~
### 永久关闭swap：

查看/etc/fstab
找到swap分区的记录
~~~~bash
...
/dev/mapper/cl-root     /                       xfs     defaults        0 0
UUID=f384615e-7c71-43b0-876c-45e8f08cfa6e /boot                   ext4    defaults        1 2
/dev/mapper/cl-home     /home                   xfs     defaults        0 0
/dev/mapper/cl-swap     swap                    swap    defaults        0 0

~~~~
把加载swap分区的那行记录注释掉即可
```bash
#/dev/mapper/cl-swap     swap                    swap    defaults        0 0
```
随后重启机器即可