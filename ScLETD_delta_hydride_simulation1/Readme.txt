1. 目录文件
	./bin				可执行文件目录
	./obj				编译中间文件目录
	./output/output1	计算结果输出目录
	./src				源文件目录
	*.dat				程序输入文件
	*.ini				程序配置文件
	makefile			编译文件
	run.sh				程序运行文件

2. 编译
	make
	生成的可执行文件在./bin目录下
	注：程序中使用了BLAS函数DGEMM，目前使用的是Intel MKL，如需要使用其他基础数学库，需加入相应的头文件和编译选项
	    头文件加入到form.c文件中，编译选项加入到makefile中

2. 运行
	sh run.sh XX
	注：程序配置文件为inputXX.ini，run.sh中设置了线程数量、MKL线程数量和线程绑定方式