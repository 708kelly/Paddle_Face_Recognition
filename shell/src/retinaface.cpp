#include "retinaface.h"
#include <paddle_api.h>
using namespace paddle::lite_api;

void ShapeProduction(const shape_t& shape) {
  for (auto i : shape) std::cout << i << std::endl;
  // return res;
}


Retinaface::Retinaface()
{
};

bool Retinaface::loadModel(std::string& aModelPath)
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
  image_tensor->Resize({mInputBatch, mInputChannel, mInputHeight, mInputWidth});
  auto image_data = image_tensor->mutable_data<float>();
  predictor->Run();  // Warmup
  // std::cout << "Warmup" << std::endl;
  return true;
};

void Retinaface::preprocess(cv::Mat& aInputImage,
                            cv::Mat& aProcessedImage)
{
  // aProcessedImage = aInputImage;
  cv::resize(aInputImage,
              aProcessedImage,
              cv::Size(mInputWidth, mInputHeight),
              0,
              0);
  aProcessedImage.convertTo(aProcessedImage, CV_32FC3, 1);
  // std::cout << aProcessedImage.rows << "  " << aProcessedImage.cols << std::endl;
};

void Retinaface::inference(cv::Mat& aProcessedImage,
                           std::vector<faceBboxStruct>& aFaceBbox)
{
  int originalWidth = aProcessedImage.size().width;
  int originalHeight = aProcessedImage.size().height;
  int originChannel = aProcessedImage.channels();
  // std::cout << "prepare input tensor." << std::endl;
  std::unique_ptr<paddle::lite_api::Tensor> image_tensor = predictor->GetInput(0);
  auto image_data = image_tensor->mutable_data<float>();
  nhwc32nc3hw(reinterpret_cast<const float *>(aProcessedImage.data),
              image_data,
              mMeanVals,
              mNormVals,
              mInputWidth,
              mInputHeight);
  // std::cout << "nhwc32nc3hw" << std::endl;
  // Inference
  double start = get_current_us();
  predictor->Run();
  double end = get_current_us();
  std::cout << "inference time:" << (end - start) / 1000.0f << std::endl;
  // // 2. inference sfaceBboxStructcores & boxes.
  
  std::vector<std::string> output_names = predictor->GetOutputNames();
  std::map<std::string, const float*> OutputMap;
  std::map<std::string, std::vector<long int>> DimMap;
  for (int i = 0; i < output_names.size(); i ++) {
    // printf("Output name[%d]: %s\n", i, output_names[i].c_str());
    auto output_tensor = predictor->GetOutput(i);
    auto output_data = output_tensor->data<float>();
    std::vector<long int> dim = {output_tensor->shape()[0], output_tensor->shape()[1], output_tensor->shape()[2]};
    DimMap[output_mp[i]] = dim;
    OutputMap[output_mp[i]] = output_data;
    // std::cout << shape_production(output_tensor->shape()) << std::endl;
  }
  // // 3. rescale & exclude.
  std::vector<faceBboxStruct> bboxCollection;
  this->generateBboxes(bboxCollection, OutputMap, DimMap, mScoreThreshold, originalHeight, originalWidth);
  

  // 4. hard|blend nms with topk.
  this->nms(bboxCollection, aFaceBbox);

  for (auto bbox: aFaceBbox) {
    std::cout << bbox.x1 << std::endl;
    std::cout << bbox.x2 << std::endl;
    std::cout << bbox.y1 << std::endl;
    std::cout << bbox.y2 << std::endl;
    std::cout << bbox.score << std::endl;
    for (int i = 0; i < 5; i++) {
      std::cout << bbox.landmarks[2*i] << " " << bbox.landmarks[2*i+1] << std::endl;
    }
  }
};

void Retinaface::generateBboxes(std::vector<faceBboxStruct> &aBboxCollection,
                                const std::map<std::string, const float*> &aOutputTensors,
                                const std::map<std::string, std::vector<long int>> &aOutputDims,
                                float aScoreThreshold,
                                float aImgHeight,
                                float aImgWidth)
{
  
  const unsigned int bboxNum = aOutputDims.at("bbox").at(1); // n = ?


  std::vector<RetinaAnchor> anchors;
  this->generateAnchors(mInputHeight, mInputWidth, anchors);

  const unsigned int numAnchors = anchors.size();
  if (numAnchors != bboxNum)
    throw std::runtime_error("mismatch numAnchors != bboxNum");

  const float *bboxesPtr = aOutputTensors.at("bbox"); // e.g (1,16800,4)
  const float *probsPtr = aOutputTensors.at("conf"); // e.g (1,16800,2) after softmax
  const float *landmarksPtr = aOutputTensors.at("landmarks");

  aBboxCollection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < numAnchors; ++i)
  {
    float conf = probsPtr[2 * i + 1];
    if (conf < aScoreThreshold) continue; // filter first.

    float priorCX = anchors.at(i).cx;
    float priorCY = anchors.at(i).cy;
    float priorSKX = anchors.at(i).s_kx;
    float priorSKY = anchors.at(i).s_ky;

    float dx = bboxesPtr[4 * i + 0];
    float dy = bboxesPtr[4 * i + 1];
    float dw = bboxesPtr[4 * i + 2];
    float dh = bboxesPtr[4 * i + 3];
    // ref: https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py
    float cx = priorCX + dx * mVariance[0] * priorSKX;
    float cy = priorCY + dy * mVariance[0] * priorSKY;
    float w = priorSKX * std::exp(dw * mVariance[1]);
    float h = priorSKY * std::exp(dh * mVariance[1]); // norm coor (0.,1.)

    faceBboxStruct box;
    box.x1 = (cx - w / 2.f) * aImgWidth;
    box.y1 = (cy - h / 2.f) * aImgHeight;
    box.x2 = (cx + w / 2.f) * aImgWidth;
    box.y2 = (cy + h / 2.f) * aImgHeight;
    box.score = conf;
    for (int i = 0; i < 5; i++) {
      float landmarkX = (priorCX + landmarksPtr[2*i] * mVariance[0] * priorSKX);
      float landmarkY = (priorCY + landmarksPtr[2*i+1] * mVariance[0] * priorSKY);
      box.landmarks[2*i] = landmarkX * aImgWidth;
      box.landmarks[2*i+1] = landmarkY * aImgHeight;
    }
    aBboxCollection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
};

void Retinaface::generateAnchors(const int aTargetHeight,
                                 const int aTargetWidth,
                                 std::vector<RetinaAnchor> &aAnchors)
{
  std::vector<std::vector<int>> featureMaps;
  for (auto step: mSteps)
  {
    featureMaps.push_back(
        {
            (int) std::ceil((float) aTargetHeight / (float) step),
            (int) std::ceil((float) aTargetWidth / (float) step)
        } // ceil
    );
  }

  aAnchors.clear();
  const int featureMapsNum = featureMaps.size();

  for (int k = 0; k < featureMapsNum; ++k)
  {
    auto featureMapsTmp = featureMaps.at(k); // e.g [640//8,640//8]
    auto minSizesTmp = mMinSizes.at(k); // e.g [8,16]
    int featureMapsHeight = featureMapsTmp.at(0);
    int featureMapsWidth = featureMapsTmp.at(1);

    for (int i = 0; i < featureMapsHeight; ++i)
    {
      for (int j = 0; j < featureMapsWidth; ++j)
      {
        for (auto min_size: minSizesTmp)
        {
          float skx = (float) min_size / (float) aTargetWidth; // e.g 16/w
          float sky = (float) min_size / (float) aTargetHeight; // e.g 16/h
          float cx = ((float) j + 0.5f) * (float) mSteps.at(k) / (float) aTargetWidth;
          float cy = ((float) i + 0.5f) * (float) mSteps.at(k) / (float) aTargetHeight;
          aAnchors.push_back(RetinaAnchor{cx, cy, skx, sky}); // without clip
        }
      }
    }
  }
};