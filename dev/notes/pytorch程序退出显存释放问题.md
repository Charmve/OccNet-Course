
1. 清空CUDA cache
``
del styler 
import torch
torch.cuda.empty_cache()
time.sleep(5)
``

2. https://blog.csdn.net/wumo1556/article/details/88413429
- fusr -v /dev/nvidia* 查看在gpu上运行的所有程序
- kill 所有连号的进程
