from signal_tools import generate_bits,bpsk_modulate,add_awgn

#1、生成10个随机比特
bits=generate_bits(10)

#2、BPSK调制：0映射为-1，1映射为+1
symbols=bpsk_modulate(bits)

#3、加入高斯白噪声
noise_std=0.3
received_signal=add_awgn(symbols,noise_std)

#4、打印结果
print("原始比特",bits)
print("BPSK符号",symbols)
print("接收信号",received_signal)

#5、接收判决
detected_bits=(received_signal>0).astype(int)
error_count=(bits!=detected_bits).sum()

print("判决比特",detected_bits)
print("错误个数",error_count)
