#ifndef MODULES_KALININ_D_VECTOR_DOT_PRODUCT_TASK_HPP_
#define MODULES_KALININ_D_VECTOR_DOT_PRODUCT_TASK_HPP_

#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace ppc {
namespace kalinin_d_vector_dot_product_seq {

template <class InOutType>
class VectorDotProduct : public ppc::core::Task {
 public:
  explicit VectorDotProduct(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<InOutType>> input_;
  InOutType dot_product_ = 0;
};

}  // namespace kalinin_d_vector_dot_product_seq
}  // namespace ppc

#endif  // MODULES_KALININ_D_VECTOR_DOT_PRODUCT_TASK_HPP_
