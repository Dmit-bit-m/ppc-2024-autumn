#include "seq/kalinin_d_vector_dot_product/include/ops_seq.hpp"

namespace ppc {
namespace kalinin_d_vector_dot_product_seq {

template <class InOutType>
bool VectorDotProduct<InOutType>::pre_processing() {
  internal_order_test();

  input_ = std::vector<std::vector<InOutType>>(2);
  for (size_t i = 0; i < input_.size(); ++i) {
    input_[i] = std::vector<InOutType>(taskData->inputs_count[i]);
    auto tmp_ptr = reinterpret_cast<InOutType*>(taskData->inputs[i]);
    for (unsigned j = 0; j < taskData->inputs_count[i]; ++j) {
      input_[i][j] = tmp_ptr[j];
    }
  }

  dot_product_ = static_cast<InOutType>(0);
  return true;
}

template <class InOutType>
bool VectorDotProduct<InOutType>::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 2) {
    return false;
  }

  bool inputs_valid = taskData->inputs_count[0] == taskData->inputs_count[1];
  bool outputs_valid = taskData->outputs_count[0] == 1;

  return inputs_valid && outputs_valid;
}

template <class InOutType>
bool VectorDotProduct<InOutType>::run() {
  internal_order_test();
  dot_product_ = std::inner_product(input_[0].begin(), input_[0].end(), input_[1].begin(), static_cast<InOutType>(0));
  return true;
}

template <class InOutType>
bool VectorDotProduct<InOutType>::post_processing() {
  internal_order_test();
  reinterpret_cast<InOutType*>(taskData->outputs[0])[0] = dot_product_;
  return true;
}

template class VectorDotProduct<int>;
template class VectorDotProduct<float>;
template class VectorDotProduct<double>;

}  // namespace kalinin_d_vector_dot_product_seq
}  // namespace ppc
