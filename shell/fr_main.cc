#include "arcface.h"
#include "retinaface.h"
#include <opencv2/opencv.hpp>

std::vector<float> getEmbedding(Retinaface* fdModel,
                                Arcface* frModel,
                                cv::Mat image,
                                int i) {
  std::string s = std::to_string(i);
  cv::Mat processedImage;
  // Dectection
  std::vector<faceBboxStruct> facebboxes;
  fdModel->preprocess(image, processedImage);
  fdModel->inference(processedImage, facebboxes);
  std::vector<cv::Mat> alignedImageContainer;
  fdModel->alignFace(processedImage, facebboxes, alignedImageContainer);
  // Recognition
  std::vector<float> embeddings;
  frModel->preprocess(alignedImageContainer[0], processedImage);
  frModel->inference(processedImage, embeddings);
  return embeddings;
}

int main(int argc, char* argv[]) {
  std::string fdModelPath = argv[1];
  std::string frModelPath = argv[2];
  std::string inputImageFileName1 = argv[3];
  std::string inputImageFileName2 = argv[4];

  Retinaface* fdModel= new Retinaface;
  bool flagFD = fdModel->loadModel(fdModelPath);
  Arcface* frModel= new Arcface;
  bool flagFR = frModel->loadModel(frModelPath);

  cv::Mat image1 = cv::imread(inputImageFileName1);
  std::vector<float> embeddings1 = getEmbedding(fdModel, frModel, image1, 0);

  cv::Mat image2 = cv::imread(inputImageFileName2);
  std::vector<float> embeddings2 = getEmbedding(fdModel, frModel, image2, 1);

  float similarity = frModel->calculateSimilarity(embeddings1, embeddings2);
  std::cout << "similarity: " << similarity << std::endl;

  return 0;
}