#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <paddle_api.h>

#include "baseModel.h"
using namespace paddle::lite_api;

class Retinaface: public FaceDetection {
  public:
    Retinaface();
    ~Retinaface();
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBbox);
  private:
    int mInputBatch = 1;
    int mInputChannel = 3;
    int mInputHeight = 640;
    int mInputWidth = 640;
    std::map<int, std::string> output_mp = {{0, "bbox"}, {1, "conf"}, {2, "landmarks"}};


    float mScoreThreshold = 0.7;
    const float mMeanVals[3] = {104.f, 117.f, 123.f}; // bgr
    const float mNormVals[3] = {1.f, 1.f, 1.f};
    const float mVariance[2] = {0.1f, 0.2f};
    std::vector<int> mSteps = {8, 16, 32};
    std::vector<std::vector<int>> mMinSizes = {
      {16,  32},
      {64,  128},
      {256, 512}
    };

    // nested classes
    struct RetinaAnchor
    {
      float cx;
      float cy;
      float s_kx;
      float s_ky;
    };

    static constexpr const unsigned int max_nms = 30000;

  private:
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
    void generateAnchors(const int aTargetHeight,
                         const int aTargetWidth,
                         std::vector<RetinaAnchor> &aAnchors);

    void generateBboxes(std::vector<faceBboxStruct> &aBboxCollection,
                        const std::map<std::string, const float*> &aOutputTensors,
                        const std::map<std::string, std::vector<long int>> &aOutputDims,
                        float aScoreThreshold,
                        float aImgHeight,
                        float aImgWidth); // rescale & exclude

}; 

#endif