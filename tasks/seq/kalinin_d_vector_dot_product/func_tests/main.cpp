#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/kalinin_d_vector_dot_product/include/ops_seq.hpp"

TEST(kalinin_d_vector_dot_product_seq, DotProduct_3_Elements) {
  std::vector<int> vec1 = {1, 2, 3};
  std::vector<int> vec2 = {4, 5, 6};
  int expected_result = 1 * 4 + 2 * 5 + 3 * 6;

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec1.data()));
  taskDataSeq->inputs_count.emplace_back(vec1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec2.data()));
  taskDataSeq->inputs_count.emplace_back(vec2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ppc::kalinin_d_vector_dot_product_seq::VectorDotProduct<int> testTask(taskDataSeq);

  ASSERT_TRUE(testTask.validation());
  ASSERT_TRUE(testTask.pre_processing());
  ASSERT_TRUE(testTask.run());
  ASSERT_TRUE(testTask.post_processing());

  ASSERT_EQ(out[0], expected_result);
}

TEST(kalinin_d_vector_dot_product_seq, DotProduct_5_Elements) {
  std::vector<int> vec1 = {2, 3, 4, 5, 6};
  std::vector<int> vec2 = {1, 2, 3, 4, 5};
  int expected_result = 2 * 1 + 3 * 2 + 4 * 3 + 5 * 4 + 6 * 5;

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec1.data()));
  taskDataSeq->inputs_count.emplace_back(vec1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec2.data()));
  taskDataSeq->inputs_count.emplace_back(vec2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ppc::kalinin_d_vector_dot_product_seq::VectorDotProduct<int> testTask(taskDataSeq);

  ASSERT_TRUE(testTask.validation());
  ASSERT_TRUE(testTask.pre_processing());
  ASSERT_TRUE(testTask.run());
  ASSERT_TRUE(testTask.post_processing());

  ASSERT_EQ(out[0], expected_result);
}

TEST(kalinin_d_vector_dot_product_seq, DotProduct_Zero_Vectors) {
  std::vector<int> vec1 = {0, 0, 0};
  std::vector<int> vec2 = {0, 0, 0};
  int expected_result = 0;

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec1.data()));
  taskDataSeq->inputs_count.emplace_back(vec1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec2.data()));
  taskDataSeq->inputs_count.emplace_back(vec2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  ppc::kalinin_d_vector_dot_product_seq::VectorDotProduct<int> testTask(taskDataSeq);

  ASSERT_TRUE(testTask.validation());
  ASSERT_TRUE(testTask.pre_processing());
  ASSERT_TRUE(testTask.run());
  ASSERT_TRUE(testTask.post_processing());

  ASSERT_EQ(out[0], expected_result);
}
