# 単純パーセプトロンによるAND回路の作成
## パーセプトロン
人口ニューロンとも呼ばれる。ニューロン（neuron）とは、生物の脳を構成する神経細胞のことである。例えば、赤い色に強く反応するニューロンや丸い形状に強く反応するニューロンなど、得意分野が異なるニューロンが多数（人間は1000億個）集まり生物は記憶や認識をしている。パーセプトロンはこれをコンピュータ内で人工的に作ったものである。
**今回は１つのパーセプトロンにAND回路を学習させる方法を解説する**。
<img width="289" alt="image.png (48.8 kB)" src="https://img.esa.io/uploads/production/attachments/18204/2022/05/11/111414/076a9413-8a3a-4352-9543-16ac3aeb303d.png">

## サンプルコード
中部大学の藤吉先生が公開されているリポジトリ。colabなので簡単に動かすことができるが数学的な解説が一切ない。
https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/Perceptron_AND.ipynb

## 数式解説
<img width="404" alt="Perceptron.drawio.png (20.7 kB)" src="https://img.esa.io/uploads/production/attachments/18204/2022/01/01/111414/83ed6709-87b9-4b52-a80c-edf0740fd184.png">

### シグモイド関数（活性化関数）
$$
 f(a) = \frac{1}{1 + e^{-a}}\tag{1}
$$
### パーセプトロン出力
$$
 o^k_1 = f(i^k_1)\tag{2}
$$
### パーセプトロン入力
$$
i^k_1 = \sum_{m=1}^2 o^{k-1}_m w^{k-1}_m\tag{3}
$$
### 損失関数
tは教師データ
$$
E = \frac{1}{2}(t - o^k_1)^2\tag{4}
$$
### 重みの更新
$\eta$は学習率
$$
w^{k-1}_1 = w^{k-1}_1 - \eta \frac{\partial E}{\partial w^{k-1}_1}\tag{5}
$$
$$
w^{k-1}_2 = w^{k-1}_2 - \eta \frac{\partial E}{\partial w^{k-1}_2}\tag{6}
$$
### 課題：重みの勾配
**式(7),(8)を証明してください。**
$$
\begin{align}
\frac{\partial E}{\partial w^{k-1}_1}\\
&= (o^k_1 - t)f'(i^k_1)o^{k-1}_1\tag{7}
\end{align}
$$

$$
\begin{align}
\frac{\partial E}{\partial w^{k-1}_2}\\
&= (o^k_1 - t)f'(i^k_1)o^{k-1}_2\tag{8}
\end{align}
$$

## 参考サイト
- https://www.scj.go.jp/omoshiro/kioku2/index.html#:~:text=%E8%84%B3%E5%85%A8%E4%BD%93%E3%81%AB%E3%81%AF%E3%80%811000,%E3%81%A8%E8%A8%80%E3%82%8F%E3%82%8C%E3%81%A6%E3%81%84%E3%81%BE%E3%81%99%E3%80%82