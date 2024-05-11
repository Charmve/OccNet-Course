<!--
 * @Author: Charmve yidazhang1@gmail.com
 * @Date: 2023-08-27 23:42:37
 * @LastEditors: Charmve yidazhang1@gmail.com
 * @LastEditTime: 2023-09-23 15:15:35
 * @FilePath: /OccNet-Course/code/README.md
 * @Version: 1.0.1
 * @Blogs: charmve.blog.csdn.net
 * @GitHub: https://github.com/Charmve
 * @Description: 
 * 
 * Copyright (c) 2023 by Charmve, All Rights Reserved. 
 * Licensed under the MIT License.
-->

```bash
git clone https://github.com/Charmve/OccNet-Course --recursive

./scripts/start_dev_docker.sh
./scripts/goto_dev_docker.sh

# env config
export PS1="[\[\e[1;32m\]\u\[\e[m\]\[\e[1;33m\]@\[\e[m\]\[\e[1;35m\]\h\[\e[m\]:\[\e[0;32m\]\w\[\e[0m\]$(__git_ps1 "\[\e[33m\](%s) \[\e[0m\]")\[\e[31m\]$(git_dirty)\[\e[0m\]] $ "
cp ./docker/dev/rcfiles/user.bash_aliases /home/$USER/.bash_aliases
cp ./docker/dev/rcfiles/user.vimrc /home/$USER/.vimrc

cd code/

```


Archievement:
- BEVFustion-TensorRT部署: https://blog.csdn.net/h904798869/article/details/132280120
- BEVFusion代码复现实践 https://blog.csdn.net/h904798869/article/details/132210022
- StreamPETR代码工程复现 https://blog.csdn.net/h904798869/article/details/135531719
- MapTR代码复现实践 https://blog.csdn.net/h904798869/article/details/132856083
