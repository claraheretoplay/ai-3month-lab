import torch

#1.创建需要计算梯度的参数
w=torch.tensor(3.0,requires_grad=True)#requires_grad=true表示需要跟踪该tensor的计算过程，并为他计算梯度
x=torch.tensor(2.0,requires_grad=True)
b=torch.tensor(1.0,requires_grad=True)

# print(w)
# print(x)
# print(b)

target=torch.tensor(10.0)

#2.前向计算：y=w*x+b
y=w*x+b

#3.计算损失：loss=(y-target)^2
loss=(y-target)**2#**是python的幂运算符，表示乘方/次方

print("y=",y.item())
print("loss=",loss.item())

#4.反向传播，自动计算梯度
loss.backward()#从损失开始，沿计算图反向传播，利用链式法则计算各参数的梯度#pytorch自动求导代替手动计算
#只负责计算，不会自动修改w,x,b，参数更新通常由优化器完成
# with torch.no_grad():
#     w-=0.01*w.grad
#     b-=0.01*b.grad

# #实际项目中通常使用优化器自动完成
# optimizer=torch.optim.SGD([w,b],lr=0.01)#SGD(Stochastic Gradient Descent)随机梯度下降,lr学习率

# optimizer.zero_grad()#清空梯度
# loss.backward()#计算梯度
# optimizer.step()#更新参数

#实际使用时还需每轮重新计算y和loss
# for epoch in range(100):
#     y=w*x+b
#     loss=(y-target)**2

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(epoch,loss.item(),w.item(),b.item())


#5.查看梯度
print("\nw.grad=",w.grad)
print("x.grad=",x.grad)
print("b.grad=",b.grad)

# #6.手动验证梯度
# y=3*2+1=7
# loss=(7-10)^2=9

# d(loss)/d(y)=2*(y-target)=-6
# d(loss)/d(w)=-6*x=-12
# d(loss)/d(x)=-6*w=-18
# d(loss)/d(b)=-6

#7.梯度会累加，下一轮计算前需要清零
w.grad.zero_()
x.grad.zero_()
b.grad.zero_()

print("\n清零后的梯度:")
print("w.grad=",w.grad)
print("x.grad=",x.grad)
print("b.grad=",b.grad)