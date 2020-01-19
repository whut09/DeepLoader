#  Deep Loader
Data processing is  important in deep learning, it take much time and is very boring to convert data formats among different deep learning frameworks (like caffe, tensorflow, PyTorch, MxNet).
The purpose of this project is to make data processing easier. 

Main features of the library are:
- [Data readers](deeploader/dataset) for many common data formats, specially for image classification
- Data batch [prefetching](deeploader/dataset/prefetcher.py) in a background thread
- Face recognition evaluation tool ,  check [run_verify](deeploader/eval/run_verify.py) for details, which can test models with LFW data
- Data adaptors for different [frameworks](deeploader/plats)
- Many [utility](deeploader/util) functions
  - Face [alignment](deeploader/util/alignment.py)
  - A [multi-step](deeploader/util/lr_schedule.py) learning rate scheduler with `decay` and `warmup` options
  - Fast image [hashing](deeploader/util/hashing) functions like P/A/D/W-Hash, and our MHash
  - [Speedometer](deeploader/util/speedmeter.py) for tracking program speed
- Tools
  - Image [labeling](deeploader/tools/label_img_page.py) tool for binary classification, it loads multiple images with a handy detail window.
  - Image [duplicate](deeploader/tools/dedup.py) removal tool to remove duplicate images
  - A video annotation format and [HTML viewer](deeploader/tools/player.md)
- ONNX
  - Fixed some bugs for convert MxNet models to ONNX [x2onnx](deeploader/x2onnx)
  - Mode forward  wrappers for different backends like (Pytorch/MxNet/ONNX)  [x2onnx](deeploader/x2onnx/backends)



# Install

```shell
pip install -e .
```



