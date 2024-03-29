set nocompatible              " be iMproved, required

set undofile " Maintain undo history between sessions
set undodir=~/.vim/undodir

set encoding=utf-8
set termencoding=utf-8
set fileencoding=utf-8
set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936

set tabstop=2
set shiftwidth=2
set softtabstop=2

set expandtab
autocmd FileType make set noexpandtab shiftwidth=4 softtabstop=0

set autoindent
set smartindent
set cindent

set hlsearch

set ruler
set autowrite
set secure
set backspace=indent,eol,start
set spell
hi clear SpellBad
hi SpellBad cterm=underline
hi SpellBad gui=undercurl

" Disable all sounds.
set belloff=all
set isfname-=: " allow g[f|F] to jump to line number

try
  source ~/.personal_vimrc
catch
  " It's OK.
endtry
