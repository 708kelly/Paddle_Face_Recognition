#include "arcface.h"
#include <cstdlib> 

#define OUTPUT_NODE_NAME "683"

Arcface::Arcface()
{
}

Arcface::~Arcface()
{
}

bool Arcface::loadModel(std::string& aModelPath)
{
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(aModelPath); // .nb
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(CPU_POWER_MODE);
  predictor =
    paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
        mobile_config);
    // Prepare for inference
  std::unique_ptr<paddle::lite_api::Tensor> image_tensor =
      predictor->GetInput(0);
  image_tensor->Resize({1, 3, height, width});
  auto image_data = image_tensor->mutable_data<float>();
  predictor->Run();  // Warmup
  return true;
}

void Arcface::preprocess(cv::Mat& origin_image, cv::Mat& resized_image)
{
  cv::resize(origin_image,
              resized_image,
              cv::Size(width, height),
              0,
              0);
  if (resized_image.channels() == 3) {
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
  } else if (resized_image.channels() == 4) {
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGRA2RGB);
  } else {
    printf("The channel size should be 4 or 3, but receive %d!\n",
            resized_image.channels());
    exit(-1);
  }
  resized_image.convertTo(resized_image, CV_32FC3, 1);
}

void Arcface::inference(cv::Mat& aProcessedImage, std::vector<float>& aEmbedding)
{
  int originalWidth = aProcessedImage.size().width;
  int originalHeight = aProcessedImage.size().height;
  int originChannel = aProcessedImage.channels();
  std::unique_ptr<paddle::lite_api::Tensor> image_tensor = predictor->GetInput(0);
  auto image_data = image_tensor->mutable_data<float>();
  nhwc32nc3hw(reinterpret_cast<const float *>(aProcessedImage.data),
              image_data,
              mean.data(),
              std.data(),
              width,
              height);
  // Inference
  double start = get_current_us();
  predictor->Run();
  double end = get_current_us();
  std::cout << "inference time:" << (end - start) / 1000.0f << std::endl;
  auto output_tensor = predictor->GetOutput(0);
  auto output_data = output_tensor->data<float>();
  
  auto output_size = shape_production(output_tensor->shape());

  std::cout << "output_size: " << output_size << std::endl;
  for (int64_t j = 0; j < output_size; j++) {
    aEmbedding.push_back(output_data[j]);
  }
}

float Arcface::calculateSimilarity(std::vector<float>& feat1, std::vector<float>& feat2) {
  float inner_product = 0.0f;
  float feat_norm1 = 0.0f;
  float feat_norm2 = 0.0f;
  for (int i = 0; i < this->mFeatureDim; ++i) {
    inner_product += feat1[i] * feat2[i];
    feat_norm1 += feat1[i] * feat1[i];
    feat_norm2 += feat2[i] * feat2[i];
	}
	return abs(inner_product) / sqrt(feat_norm1) / sqrt(feat_norm2);
}