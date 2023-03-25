#include "baseModel.h"
#include "faceProcess.h"

BaseModel::BaseModel()
{
}

BaseModel::~BaseModel()
{
}

int64_t BaseModel::get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

void BaseModel::nhwc32nc3hw(const float *src,
                 float *dst,
                 const float *mean,
                 const float *std,
                 int width,
                 int height) {
  int size = height * width;
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) / std[0];
    *(dst_c1++) = (*(src++) - mean[1]) / std[1];
    *(dst_c2++) = (*(src++) - mean[2]) / std[2];
  }
}

int64_t BaseModel::shape_production(std::vector<int64_t> shape) {
  int64_t s = 1;
  for (int64_t dim : shape) {
    s *= dim;
  }
  return s;
}

FaceDetection::FaceDetection()
{
}

FaceDetection::~FaceDetection()
{
}

bool FaceDetection::loadModel(std::string& aModelPath)
{
  return false;
}

void FaceDetection::preprocess(cv::Mat& aInputImage,
                               cv::Mat& aProcessedImage)
{
}
                          
void FaceDetection::inference(cv::Mat& aProcessedImage,
                              std::vector<faceBboxStruct>& aFaceBbox)
{
}

float FaceDetection::getIOU(faceBboxStruct& box1, faceBboxStruct& box2)
{
  float innerX1 = box1.x1 > box2.x1 ? box1.x1 : box2.x1;
  float innerY1 = box1.y1 > box2.y1 ? box1.y1 : box2.y1;
  float innerX2 = box1.x2 < box2.x2 ? box1.x2 : box2.x2;
  float innerY2 = box1.y2 < box2.y2 ? box1.y2 : box2.y2;
  float innerH = innerY2 - innerY1 + 1.0f;
  float innerW = innerX2 - innerX1 + 1.0f;
  float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

  if (innerH <= 0.f || innerW <= 0.f) {
    return std::numeric_limits<float>::min();
  }
  float innerArea = innerH * innerW;
  return innerArea / (area1 + area2 - innerArea);
}

void FaceDetection::nms(std::vector<faceBboxStruct> &aInputBboxes,
                        std::vector<faceBboxStruct> &aOutputBboxes,
                        float aIouThreshold,
                        unsigned int topk)
{
  std::sort(aInputBboxes.begin(), aInputBboxes.end(),
            [](const faceBboxStruct &a, const faceBboxStruct &b)
            { return a.score > b.score; });
  const unsigned int boxNum = aInputBboxes.size();
  std::vector<int> merged (boxNum, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < boxNum; ++i)
  {
    if (merged[i]) continue;
    std::vector<faceBboxStruct> buf;

    buf.push_back(aInputBboxes[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < boxNum; ++j)
    {
      if (merged[j]) continue;

      float iou = getIOU(aInputBboxes[i], aInputBboxes[j]);

      if (iou > aIouThreshold)
      {
        merged[j] = 1;
        buf.push_back(aInputBboxes[j]);
      }
    }
    aOutputBboxes.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
};

void FaceDetection::alignFace(cv::Mat& aProcessedImage,
                              std::vector<faceBboxStruct>& aFaceBboxes,
                              std::vector<cv::Mat>& aAlignedImageContainer)
{
  float srcPoints[5][2] = {
    {30.2946, 51.6963},
    {65.5318, 51.5014},
    {48.0252, 71.7366},
    {33.5493, 92.3655},
    {62.7299, 92.2041}
  };
  float detectPoints[5][2];
  for (faceBboxStruct bbox: aFaceBboxes) {
    memcpy(detectPoints[0], bbox.landmarks, 10*sizeof(float));
    cv::Mat srcMat(5, 2, CV_32FC1, srcPoints);
    memcpy(srcMat.data, srcPoints, 10*sizeof(float));
    cv::Mat dstMat(5, 2, CV_32FC1, detectPoints);
    memcpy(dstMat.data, detectPoints, 10*sizeof(float));
    cv::Mat M = FacePreprocess::similarTransform(dstMat, srcMat);
    cv::Mat wrapImg;
    cv::warpAffine(aProcessedImage, wrapImg, M, cv::Size(112, 112));
    aAlignedImageContainer.push_back(wrapImg);
  }
};



FaceRecognition::FaceRecognition()
{
}

FaceRecognition::~FaceRecognition()
{
}

bool FaceRecognition::loadModel(std::string& aModelPath)
{
  return false;
}

void FaceRecognition::preprocess(cv::Mat& aInputImage,
                                 cv::Mat& aProcessedImage)
{
}

void inference(cv::Mat& aProcessedImage,
               std::vector<float>& aEmbedding)
{
}

float calculateSimilarity(std::vector<float>& aEmbedding1,
                          std::vector<float>& aEmbedding2)
{
  return 0.0f;
}
