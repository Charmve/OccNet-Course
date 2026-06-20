# !/bin/bash  

export PS1='[\[\e[1;32m\]\u\[\e[m\]\[\e[1;33m\]@\[\e[m\]\[\e[1;35m\]\h\[\e[m\]:\[\e[0;32m\]\w\[\e[0m\]$(__git_ps1 "\[\e[33m\](%s) \[\e[0m\]")\[\e[31m\]$(git_dirty)\[\e[0m\]] $ '
export LANG=C.UTF-8

apt update
apt install -y vim
apt install git-lfs
apt install zip unzip

cp /hosthome/.gitconfig ~/.gitconfig

git config --global --add safe.directory /maiwei

cp ./docker/dev/rcfiles/user.bash_aliases ~/.bash_aliases
cp ./docker/dev/rcfiles/user.vimrc ~/.vimrc
