## 689 Project plan:

### Notes:
1. Finalized datasets: Most commonly used across papers/implementations. Should be a good starting point.
	- [UCF 101](http://crcv.ucf.edu/data/UCF101.php)
	- Youtube MSR corpus - already downloaded.

2. 1 and 2 below are in Caffe. 3 & 4 are in Theano.

3.  Implementation by Thursday night.
4. Training till Saturday.
5. Analyze Results - Sunday.

### Implementations:
1. [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video)
	- [Paper](http://arxiv.org/abs/1411.4389)
	- [Code](https://github.com/LisaAnne/lisa-caffe-public/tree/lstm_video_deploy)
	- [Data](http://crcv.ucf.edu/data/UCF101.php)
	- [Pre-trained Models](https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video_weights.html)
	- Comments: Uses flow - can train on another data set (the youtube one) for a better model.
	- Additional : Donahue's [website](http://jeffdonahue.com/lrcn/)

2. [Sequence to Sequence Video to Text](https://www.cs.utexas.edu/~vsub/s2vt.html#code)
	- [Paper](https://arxiv.org/pdf/1412.4729v3.pdf)
	- [Code](https://github.com/vsubhashini/caffe/tree/recurrent/examples/s2vt)
	- Data - already downloaded (the youtube) - can also use the one from [1] above.
	- [Pre-trained Model](https://www.dropbox.com/s/wn6k2oqurxzt6e2/s2s_vgg_pstream_allvocab_fac2_iter_16000.caffemodel?dl=1)
	-  Comments: Uses flow - can also train on the UFC-101 dataset above.

3. [Describing Videos by Exploiting Temporal Structure](http://arxiv.org/abs/1502.08029)
	- [Paper](http://arxiv.org/pdf/1502.08029v4.pdf)
	- [Code](https://github.com/yaoli/arctic-capgen-vid)
	- [Data](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/yaoli/youtube2text_iccv15.zip)
	- Comments - attention based models. But uses 3D-CNNs which might be hard to implement.

4. [Action Recognition using Visual Attention](http://shikharsharma.com/projects/action-recognition-attention/)
: [Alternate link](http://www.gitxiv.com/posts/xdxtvLF2angdj9BKW/action-recognition-using-visual-attention)
	- [Paper](https://arxiv.org/pdf/1511.04119v3.pdf)
	- [Code](https://github.com/kracwarlock/action-recognition-visual-attention)
	- Data - UFC 101 same as [1] above.
	- Comments - attention based models.
