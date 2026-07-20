import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    #sigmoid函数，限制输入范围避免溢出
    z=np.clip(z,-50,50)
    return 1/(1+np.exp(-z))

def train_logistic_regression(X,y,learning_rate=0.1,epochs=1000):
    #手写逻辑回归训练
    n_samples,n_features=X.shape

    w=np.zeros((n_features,1))#n是每个特征一个权重，当且仅当n_features=1时，此处为(1,1)
    b=0.0

    y=y.reshape(-1,1)

    for _ in range(epochs):
        #计算预测概率
        score=X@w+b
        probability=sigmoid(score)

        #计算梯度
        error=probability-y
        dw=(X.T@error)/n_samples
        db=np.mean(error)

        #更新参数
        w=w-learning_rate*dw
        b=b-learning_rate*db

    return w,b

def predict_logistic(X,w,b):
    #逻辑回归预测
    probability=sigmoid(X@w+b)
    return(probability>=0.5).astype(int).ravel()

def train_svm(X,y,learning_rate=0.01,epochs=1000):
    #手写SVM hinge loss训练
    n_samples,n_features=X.shape

    w=np.zeros((n_features,1))
    b=0.0

    y=y.reshape(-1,1)

    for _ in range(epochs):
        #计算分类得分
        score=X@w+b
        margin=y*score

        #只有margin<1的样本产生梯度
        active=(margin<1).astype(float)

        #hinge loss梯度
        dw=-np.mean(active*y*X,axis=0).reshape(n_features,1)
        db=-np.mean(active*y)

        #更新参数
        w=w-learning_rate*dw
        b=b-learning_rate*db

    return w,b

def predict_svm(X,w,b):
    #svm预测
    score=X@w+b
    return np.where(score>=0,1,-1).ravel()

#固定随机种子
np.random.seed(42)

#测试的噪声标准差
noise_stds=[0.2,0.5,1.0]

logistic_accuracies=[]
svm_accuracies=[]

for noise_std in noise_stds:
    #生成随机比特
    n_samples=1000
    bits=np.random.randint(0,2,size=n_samples)

    #BPSK映射：0->-1,1->+1
    symbols=2*bits-1

    #添加AWGN高斯噪声
    noise=noise_std*np.random.randn(n_samples)
    received=symbols+noise

    #输入数据
    X=received.reshape(-1,1)

    #逻辑回归使用0/1标签
    logistic_y=bits

    #svm使用-1/+1标签
    svm_y=symbols

    #训练逻辑回归
    logistic_w,logistic_b=train_logistic_regression(X,logistic_y)
    logistic_pred=predict_logistic(X,logistic_w,logistic_b)
    logistic_accuracy=np.mean(logistic_pred==bits)

    #训练svm
    svm_w,svm_b=train_svm(X,svm_y)
    svm_pred=predict_svm(X,svm_w,svm_b)
    svm_accuracy=np.mean(svm_pred==symbols)

    #保存准确率
    logistic_accuracies.append(logistic_accuracy)
    svm_accuracies.append(svm_accuracy)

    print(f"noise_std={noise_std}")#f""表示这是一个格式化字符串，变量放在{}中
    print(f"Logistic Regression accuracy:{logistic_accuracy:.4f}")
    print(f"SVM accuracy:{svm_accuracy:.4f}")#.4f:保留小数点后4位，按浮点数格式显示#若无点，4代表最小宽度，即没指定小数精度，浮点数格式默认保留6位小数
    print()#打印空行

#绘制准确率曲线
plt.plot(noise_stds,logistic_accuracies,marker="o",label="Logistic Regression")#圆形标记
plt.plot(noise_stds,svm_accuracies,marker="s",label="svm")#方形标记

plt.xlabel("noise standard deviation")
plt.ylabel("accuracy")
plt.title("model comparison under different noise levels")
plt.ylim(0.5,1.05)
plt.grid(True)
plt.legend()#在图中显示图例用于区分两条曲线
plt.show()
