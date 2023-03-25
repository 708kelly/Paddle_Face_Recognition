#ifndef _BASEMODEL_H_
#define _BASEMODEL_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <paddle_api.h>
#include <sys/time.h>

class BaseModel
{
  public:
    const int CPU_THREAD_NUM = 1;
    const paddle::lite_api::PowerMode CPU_POWER_MODE =
        paddle::lite_api::PowerMode::LITE_POWER_NO_BIND;
    ~BaseModel();
    virtual bool loadModel(std::string& aModelPath) = 0;
    virtual void preprocess(cv::Mat& aInputImage, cv::Mat& aProcessedImage) = 0;
    int64_t get_current_us();
    void nhwc32nc3hw(const float *src,
                 float *dst,
                 const float *mean,
                 const float *std,
                 int width,
                 int height);
    int64_t shape_production(std::vector<int64_t> shape);
  protected:
    BaseModel();
};

struct faceBboxStruct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  float landmarks[10];
};


class FaceDetection : public BaseModel
{
  public:
    FaceDetection();
    ~FaceDetection();
   
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBbox);
    void alignFace(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBboxes,
                   std::vector<cv::Mat>& aAlignedImageContainer);

  protected:
    void nms(std::vector<faceBboxStruct>& input,
             std::vector<faceBboxStruct>& output,
             float iouThreshold = 0.6,
             unsigned int topk = 300);
    
    float getIOU(faceBboxStruct& box1, faceBboxStruct& box2);
};

class FaceRecognition : public BaseModel
{
  public:
    FaceRecognition();
    ~FaceRecognition();
    int WARMUP_COUNT = 1;
    int REPEAT_COUNT = 5;
   
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<float>& aEmbedding);
    float calculateSimilarity(std::vector<float>& aEmbedding1,
                              std::vector<float>& aEmbedding2);
  protected:
    const int mFeatureDim = 512;
    const int mInputSize = 112;
};

#endif 