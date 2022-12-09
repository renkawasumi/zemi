# MLPによる多クラス分類（MNIST）
## ブロック図
[MLPによる多クラス分類（MNIST）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/mlp_mnist.ipynb)のコードをブロック図にした。

<img width="694" alt="maluti_MLP.drawio (4).png (47.2 kB)" src="https://img.esa.io/uploads/production/attachments/18204/2022/01/02/111414/b1fdfd4b-6052-4857-8857-2b09ae44d8f4.png">

X：入力
D：全結合層
A：シグモイド関数
S：ソフトマックス関数
C：交差エントロピー
B：バッチ

## ソフトマックス関数
$$
y_k=\frac{exp(h_{5,k})}{\sum_{i=1}^n exp(h_{5,i})}\tag{1}
$$
## 交差エントロピー
$t_k$は教師データ
$$
E = -\sum_{k=1}^n t_k log(y_k)\tag{2}
$$
## 重みの更新
$\eta$は学習率
$$
w_3 = w_3 - \eta \frac{\partial E}{\partial w_3}\tag{3}
$$
$$
w_2 = w_2 - \eta \frac{\partial E}{\partial w_2}\tag{4}
$$
$$
w_1 = w_1 - \eta \frac{\partial E}{\partial w_1}\tag{5}
$$

## 重みの勾配
$$
\begin{align}
\frac{\partial E}{\partial w_{3}}\\
&= \frac{\partial E}{\partial h_{5}}\frac{\partial h_5}{\partial w_{3}}\\
&= (y - t)h_4\tag{10}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial w_{2}}\\
&= \frac{\partial E}{\partial h_{3}}\frac{\partial h_3}{\partial w_{2}}\\
&= \frac{\partial E}{\partial h_{4}}\frac{\partial h_4}{\partial h_{3}}\frac{\partial h_3}{\partial w_2}\\
&= \frac{\partial E}{\partial h_{5}}\frac{\partial h_5}{\partial h_{4}}\frac{\partial h_4}{\partial h_3}\frac{\partial h_3}{\partial w_2}\\
&= (y - t)w_3A'(h_3)h_2\tag{11}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial w_{1}}\\
&= \frac{\partial E}{\partial h_{1}}\frac{\partial h_1}{\partial w_{1}}\\
&= \frac{\partial E}{\partial h_{2}}\frac{\partial h_2}{\partial h_{1}}\frac{\partial h_1}{\partial w_1}\\
&= \frac{\partial E}{\partial h_{4}}\frac{\partial h_4}{\partial h_{3}}\frac{\partial h_3}{\partial h_{2}}\frac{\partial h_2}{\partial h_1}\frac{\partial h_1}{\partial w_1}\\
&= \frac{\partial E}{\partial h_{5}}\frac{\partial h_5}{\partial h_{4}}\frac{\partial h_4}{\partial h_{3}}\frac{\partial h_3}{\partial h_{2}}\frac{\partial h_2}{\partial h_1}\frac{\partial h_1}{\partial w_1}\\
&= (y - t)w_3A'(h_3)w_2A'(h_1)x\tag{12}
\end{align}
$$

## 課題：$\frac{\partial E}{\partial h_{5}}$の導出
(1)式を(2)式に代入する
$$
E = -\sum_{k=1}^n t_k log(\frac{exp(h_{5,k})}{\sum_{i=1}^n exp(h_{5,i})})\tag{6}
$$
**式(7),(8),(9)を証明してください。計算しやすいように$n=2$。教師データはOne-hot表現のため$\sum_{k=1}^2t_k=1$となる。**
$$
\begin{align}
E = -(t_1h_{5,1}+t_2h_{5,2})+log(exp(h_{5,1})+exp(h_{5,2}))\tag{7}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial h_{5,1}} = y_1 - t_1\tag{8}
\end{align}
$$
$$
\begin{align}
\frac{\partial E}{\partial h_{5,2}}
&= y_2 - t_2\tag{9}
\end{align}
$$
式(8)と(9)をまとめると
$$
\begin{align}
\frac{\partial E}{\partial h_5}
&= y - t\tag{10}
\end{align}
$$