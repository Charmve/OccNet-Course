set number

" Insert namespace maiwei
ab ns namespace maiwei {}

" 定义快捷键的前缀，即<Leader>
let mapleader=";"

" 开启文件类型侦测
filetype on
" 根据侦测到的不同类型加载对应的插件
filetype plugin on

" 定义快捷键到行首和行尾
nmap LB 0
nmap LE $
" 设置快捷键将选中文本块复制至系统剪贴板
vnoremap <Leader>y "+y
" 设置快捷键将系统剪贴板内容粘贴至 vim
nmap <Leader>p "+p
" 定义快捷键关闭当前分割窗口
nmap <Leader>q :q<CR>
" 定义快捷键保存当前窗口内容
nmap <Leader>w :w<CR>
" 定义快捷键保存所有窗口内容并退出 vim
nmap <Leader>WQ :wa<CR>:q<CR>
" 不做任何保存，直接退出 vim
nmap <Leader>Q :qa!<CR>
" 依次遍历子窗口
nnoremap nw <C-W><C-W>
" 跳转至右方的窗口
nnoremap <Leader>lw <C-W>l
" 跳转至左方的窗口
nnoremap <Leader>hw <C-W>h
" 跳转至上方的子窗口
nnoremap <Leader>kw <C-W>k
" 跳转至下方的子窗口
nnoremap <Leader>jw <C-W>j
" FormatCode
nnoremap <Leader>f :FormatCode<CR>

nnoremap <Leader>jw <C-W>j
" 定义快捷键在结对符之间跳转
nmap <Leader>M %

" Add Header
nmap <Leader>ah :HeaderguardAdd<cr>


" 让配置变更立即生效
autocmd BufWritePost $MYVIMRC source $MYVIMRC

" vundle 环境设置
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
" vundle 管理的插件列表必须位于 vundle#begin() 和 vundle#end() 之间

call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'altercation/vim-colors-solarized'
Plugin 'Lokaltog/vim-powerline'
Plugin 'octol/vim-cpp-enhanced-highlight'
Plugin 'Valloric/YouCompleteMe'
Plugin 'google/vim-maktaba'
Plugin 'google/vim-codefmt'
Plugin 'google/vim-glaive'
Plugin 'taglist.vim'
" 插件列表结束
call vundle#end()

filetype plugin indent on

syntax enable
syntax on
set background=light
"set background=dark

let g:solarized_termcolors=256
colorscheme solarized

" 配色方案
"colorscheme solarized
"colorscheme molokai
"colorscheme phd

" 总是显示状态栏
set laststatus=2
" 显示光标当前位置
set ruler
" 高亮显示当前行/列
set cursorline
"set cursorcolumn
" 高亮显示搜索结果
set hlsearch

" set guifont=YaHei\ Consolas\ Hybrid\ 11.5

" 自适应不同语言的智能缩进
filetype indent on
" 将制表符扩展为空格
set expandtab
" 设置编辑时制表符占用空格数
set tabstop=2
" 设置格式化时制表符占用空格数
set shiftwidth=2

" Set ClangFormat with shorkey CTRL+F
nmap <C-F> :ClangFormat<cr>

" Set YouCompleteMe follow https://www.jianshu.com/p/d908ce81017a
let g:ycm_server_python_interpreter='/usr/bin/python3'
let g:ycm_global_ycm_extra_conf='~/.vim/.ycm_extra_conf.py'

" Jump to declare 
nmap <C-L> :YcmCompleter GoToDefinitionElseDeclaration<CR>

" Mark line
set colorcolumn=80

" Highlight cpp
let g:cpp_class_scope_highlight=1
let g:cpp_member_variable_highlight=1
let g:cpp_class_decl_highlight=1
let g:cpp_experimental_simple_template_highlight=1
let g:cpp_experimental_template_highlight=1
let g:cpp_no_function_highlight=1

"cscope
if has("cscope")
   " set to 1 if you want the reverse search order.
   set csto=1

   " add any cscope database in current directory
   cs add /qcraft/cscope.out 

   nmap <C-M> :cs find c <C-R>=expand("<cword>")<CR><CR>
endif

" taglist
let Tlist_Show_One_File=1
let Tlist_Exit_OnlyWindow=1
let Tlist_SHow_Menu=1
let Tlist_File_Fold_Auto_Close=1
map <silent> <F9> :TlistToggle<cr>
