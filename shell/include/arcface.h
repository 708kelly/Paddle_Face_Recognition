#ifndef _ARCFACE_H_
#define _ARCFACE_H_

#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <paddle_api.h>

#include "baseModel.h"


class Arcface : public FaceRecognition
{
  public:
    Arcface();
    ~Arcface();
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<float>& aEmbedding);
    float calculateSimilarity(std::vector<float>& aEmbedding1,
                              std::vector<float>& aEmbedding2);

  private:
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
    int width = 112;
    int height = 112;
    std::vector<float> std = {127.5f , 127.5f , 127.5f }; 
    std::vector<float> mean = {127.5f , 127.5f , 127.5f };
};

#endif