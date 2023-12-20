#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main() {
    // Load the trained model
    Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::load("/home/ahmed/coding/opencvprojects/handwrittingrecognition/handwrittenDigitsRecognition/data/model.xml");

    // Load the image you want to classify
    Mat image = imread("/home/ahmed/coding/opencvprojects/handwrittingrecognition/handwrittenDigitsRecognition/3.png", cv::IMREAD_GRAYSCALE);  // Assuming the image is grayscale

    cv::resize(image,image,cv::Size(28,28));
    // Flatten the image
    Mat flattenedImage = image.reshape(1, 1);
    Mat input;
    flattenedImage.convertTo(input, CV_32F);

    // Perform prediction
    Mat output;
    mlp->predict(input, output);


    cout<<output<<std::endl;
    // Find the class with the highest probability
    Point classIdPoint;
    double confidence;
    minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

    int predictedClass = classIdPoint.x;

    // Display the result
    cout << "Predicted class: " << predictedClass << " with confidence: " << confidence << std::endl;

    return 0;
}
//~test.cpp
