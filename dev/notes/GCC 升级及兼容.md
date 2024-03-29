## GCC 降级

最近因为服务器系统升级为ubuntu20.04，该系统自带的gcc版本是9.3。而cuda10.1不支持gcc-9。所以需要降级。

首先，安装gcc-7：

```bash
sudo apt-get install gcc-7 g++-7
```

设置gcc版本的优先级：

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1
```

查看优先级：

```bash
sudo update-alternatives --display gcc
```

同理设置g++：

```bash
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1
sudo update-alternatives --display g++
```