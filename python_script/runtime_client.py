#!/usr/bin/python3
# 文件名：client.py

# 导入 socket、sys 模块
import socket
from time import ctime
import sys
import os
import time

# host = socket.gethostname()
host = "172.16.1.14"
port = 6666
BUFSIZ = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect((host, port))

while True:
    data = input(">>").strip()
    if not data:
        break
    data = str(len(data)) + '\n' + data
    print(data)
    s.send(data.encode('utf-8'))

    loadabel_file = open('inception_v1-nv_large-direct-quant.nvdla', 'rb')
    data = loadabel_file.read()
    data = str(len(data)) + '\n' + str(data)
    s.send(data.encode('utf-8'))

    recv_data = s.recv(BUFSIZ)
    if not recv_data:
        break
    data_len = int(recv_data)
    print(data_len)
    data = s.recv(data_len)
    print(data)

# msg = 'GET_WELCOME'
# msg_len = len(msg)
# print(msg_len)
# msg = str(msg_len) + '\n' + msg

# s.send(msg.encode('utf-8'))

# # 接收小于 1024 字节的数据
# msg = s.recv()

# time.sleep(10)
# s.close()

# print (msg.decode('utf-8'))