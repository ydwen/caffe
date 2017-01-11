#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */
template <typename Dtype>
class NormalizeLayer : public Layer<Dtype> {
 public:
  explicit NormalizeLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sum_multiplier_, norm_, squared_;
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
