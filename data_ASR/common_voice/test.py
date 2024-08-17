import time

for i in range(3):
    print('正在处理第{}条记录...'.format(i+1), i, end="\r",flush=True)
    time.sleep(1)
print('处理完成')
print('处理完成')