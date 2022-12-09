# MLPによるXOR回路の作成
<img width="684" alt="MLP.drawio.png (64.1 kB)" src="https://img.esa.io/uploads/production/attachments/18204/2022/01/01/111414/ffe18f8b-4fb0-404a-b906-adb596d01fee.png">

[MLPによるXOR回路の作成](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/MLP_XOR.ipynb)

## シグモイド関数
$$
 f(a) = \frac{1}{1 + e^{-a}}\tag{1}
$$
## ノード出力
$$
 o^k_1 = f(i^k_1)\tag{2}
$$
$$
 o^k_2 = f(i^k_2)\tag{3}
$$
$$
 o^{k+1}_1 = f(i^{k+1}_1)\tag{4}
$$
## ノード入力
$$
i^k_j = \sum_{m=1}^2 o^{k-1}_m w^{k-1,k}_{m,j}\tag{5}
$$
$$
i^{k+1}_1 = \sum_{n=1}^2 o^{k}_n w^{k,k+1}_{n,1}\tag{6}
$$
## 損失関数
tは教師データ
$$
E = \frac{1}{2}(t - o^{k+1}_1)^2\tag{7}
$$
## 重みの更新
$\eta$は学習率
### 中間層 - 出力層
$$
w^{k,k+1}_{1,1} = w^{k,k+1}_{1,1} - \eta \frac{\partial E}{\partial w^{k,k+1}_{1,1}}\tag{8}
$$
$$
w^{k,k+1}_{2,1} = w^{k,k+1}_{2,1} - \eta \frac{\partial E}{\partial w^{k,k+1}_{2,1}}\tag{9}
$$
### 入力層 - 中間層
$$
w^{k-1,k}_{1,1} = w^{k-1,k}_{1,1} - \eta \frac{\partial E}{\partial w^{k-1,k}_{1,1}}\tag{10}
$$
$$
w^{k-1,k}_{1,2} = w^{k-1,k}_{1,2} - \eta \frac{\partial E}{\partial w^{k-1,k}_{1,2}}\tag{11}
$$
$$
w^{k-1,k}_{2,1} = w^{k-1,k}_{2,1} - \eta \frac{\partial E}{\partial w^{k-1,k}_{2,1}}\tag{12}
$$
$$
w^{k-1,k}_{2,2} = w^{k-1,k}_{2,2} - \eta \frac{\partial E}{\partial w^{k-1,k}_{2,2}}\tag{13}
$$

## 課題：重みの勾配
**式(16),(18)を証明してください。**
### 中間層 - 出力層
$$
\begin{align}
\frac{\partial E}{\partial w^{k,k+1}_{1,1}}\\
&= \frac{\partial E}{\partial i^{k+1}_1} \frac{\partial i^{k+1}_1}{\partial w^{k,k+1}_{1,1}}\\
&= \frac{\partial E}{\partial o^{k+1}_1} \frac{\partial o^{k+1}_1}{\partial i^{k+1}_1} \frac{\partial i^{k+1}_1}{\partial w^{k,k+1}_{1,1}}\\
&= (o^{k+1}_1 - t)f'(i^{k+1}_1)o^{k}_1\tag{14}
\end{align}
 $$
$$
\begin{align}
\frac{\partial E}{\partial w^{k,k+1}_{2,1}}\\
&= (o^{k+1}_1 - t)f'(i^{k+1}_1)o^{k}_2\tag{15}
\end{align}
 $$
### 入力層 - 中間層
$$
\begin{align}
\frac{\partial E}{\partial w^{k-1,k}_{1,1}}\\
&= (o^{k+1}_1 - t)f'(i^{k+1}_1)w^{k,k+1}_{1,1}f'(i^{k}_1)o^{k-1}_1\tag{16}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial w^{k-1,k}_{2,1}}\\
&= (o^{k+1}_1 - t)f'(i^{k+1}_1)w^{k,k+1}_{1,1}f'(i^{k}_1)o^{k-1}_2\tag{17}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial w^{k-1,k}_{1,2}}\\
&= (o^{k+1}_1 - t)f'(i^{k+1}_1)w^{k,k+1}_{2,1}f'(i^{k}_2)o^{k-1}_1\tag{18}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial w^{k-1,k}_{2,2}}\\
&= (o^{k+1}_1 - t)f'(i^{k+1}_1)w^{k,k+1}_{2,1}f'(i^{k}_2)o^{k-1}_2\tag{19}
\end{align}
$$