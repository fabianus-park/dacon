import this_configuation

import time
# from time import process_time
start = time.process_time()
start2 = time.time()

import datetime

now = datetime.datetime.now()
# print(now)

nowDate = now.strftime('%Y-%m-%d')
# print(nowDate)

nowTime = now.strftime('%H:%M:%S')
# print(nowTime)

#nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
nowDatetime = now.strftime('%Y%m%d%H%M%S')
print(now, nowDate, nowTime, nowDatetime)

nowDatetime = now.strftime('%Y%m%d%H%M%S')

# time.sleep(5)

end = time.process_time()
print( end - start, start, end )

print( time.time() - start2, start2 )

# path = "./"
# path = "./seoul_landmark/dataset/"
# file_list = os.listdir(path)
# print(file_list)

# path = "./dataset/*"
# file_list = glob.glob(path)
# print(file_list)



# import this_predict
# from this_predict import *


if __name__ == "__main__":
    print(now)
